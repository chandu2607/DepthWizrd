import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import HeightEstimator, Sample
from .fusion_head import SmallFusionUNet

# Precalculated DFC2023 train bin counts from Phase 21:
# Bin 0 (<10m): 1227, Bin 1 (10-20m): 440, Bin 2 (20-30m): 150, Bin 3 (30-40m): 114, Bin 4 (>=40m): 61
BIN_COUNTS = np.array([1227.0, 440.0, 150.0, 114.0, 61.0])
# Moderate square-root based relative weights:
BIN_REL_WEIGHTS = 1.0 / np.sqrt(BIN_COUNTS)
# Normalize so that the average weight is 1.0
BIN_WEIGHTS = BIN_REL_WEIGHTS / np.mean(BIN_REL_WEIGHTS)

REGIME_BASES = np.array([5.0, 15.0, 25.0, 35.0, 45.0], dtype=np.float32)

class BuildingConditionedHeightNet(nn.Module):
    """First neural building-conditioned height model.
    
    1. Footprint branch: SmallFusionUNet predicts building mask probability map.
    2. Feature extractor: SmallFusionUNet outputs dense representation (feat_map).
    3. Object Pooling: connected components on predicted mask to extract geometries and local depth.
    4. MLP Heads: height regime probability (5 bins) and continuous log-scale residual.
    """
    def __init__(self, w: int = 24, C_feat: int = 16, num_regimes: int = 3):
        super().__init__()
        # Backbone is a UNet mapping [RGB(3)+Depth(1)] -> [FeatMap(C_feat) + MaskLogits(1)]
        self.backbone = SmallFusionUNet(w=w, in_channels=4, out_channels=C_feat + 1)
        self.C_feat = C_feat
        self.num_regimes = num_regimes
        
        # Features pooled: 7 geometry + 9 depth stats + 3 context + C_feat CNN = 19 + C_feat features
        in_dim = 19 + C_feat
        
        # Gating network MLP
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 3) # 3 experts
        )
        
        # Lightweight expert MLP heads
        self.expert_1 = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        self.expert_2 = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        self.expert_3 = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        
        # Trainable scaling parameters for High-Rise expert base anchor (E3)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        
        # Zero-initialize the final layers of experts so they start near baseline predictions
        nn.init.zeros_(self.expert_1[-1].weight)
        nn.init.zeros_(self.expert_1[-1].bias)
        nn.init.zeros_(self.expert_2[-1].weight)
        nn.init.zeros_(self.expert_2[-1].bias)
        nn.init.zeros_(self.expert_3[-1].weight)
        nn.init.zeros_(self.expert_3[-1].bias)
        
        # Register buffers for relative depth normalization
        self.register_buffer("d_mean", torch.tensor(0.0))
        self.register_buffer("d_std", torch.tensor(1.0))

    def forward(self, x, depth_raw=None, gt_h=None, device="cpu"):
        # x: [B, 4, H, W] (RGB + norm depth)
        # depth_raw: [B, H, W] (unnormalized depth map in float32)
        # gt_h: [B, H, W] (ground truth DSM in meters, optional)
        B, _, H, W = x.shape
        out = self.backbone(x) # [B, C_feat + 1, H, W]
        
        feat_map = out[:, :self.C_feat, :, :].float()
        mask_logits = out[:, self.C_feat:, :, :].squeeze(1) # [B, H, W]
        probs = torch.sigmoid(mask_logits)
        
        # Teacher forcing: use ground-truth mask for connected components during training if available
        if self.training and gt_h is not None:
            mask_np = (gt_h > 2.0).detach().cpu().numpy().astype(np.uint8)
        else:
            mask_np = (probs > 0.5).detach().cpu().numpy().astype(np.uint8)
        
        building_preds = []
        building_targets = []
        building_regimes = []
        building_weights = []
        
        # We perform object-level pooling per batch item
        for b in range(B):
            mask_b = mask_np[b]
            tile_density = float(mask_b.mean())
            
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_b, connectivity=8)
            
            # Find and sort components by area (>= 16 pixels)
            valid_comps = []
            for i in range(1, n):
                area_px = stats[i, cv2.CC_STAT_AREA]
                if area_px >= 16:
                    valid_comps.append((area_px, i))
            valid_comps.sort(key=lambda x: x[0], reverse=True)
            # Cap at top 25 components to avoid startup noise bottleneck
            valid_comps = valid_comps[:25]
            
            areas = [vc[0] for vc in valid_comps]
            tile_avg_building_area = float(np.mean(areas)) * 0.25 if len(areas) > 0 else 0.0
            tile_n_buildings = len(areas)
            
            # Pre-transfer depth and ground truth maps to CPU once per tile to avoid inner loop CPU-GPU transfers
            if depth_raw is not None:
                depth_np = depth_raw[b].detach().cpu().numpy().astype(np.float64)
                # Normalize relative depth to standard unit scale using registered buffers
                d_mean_val = float(self.d_mean.item())
                d_std_val = float(self.d_std.item())
                depth_np = (depth_np - d_mean_val) / (d_std_val + 1e-6)
            else:
                depth_np = None
            gt_np = gt_h[b].detach().cpu().numpy() if gt_h is not None else None
            
            # CPU gradient magnitude for local depth variation
            if depth_np is not None:
                dy, dx = np.gradient(depth_np)
                grad_mag = np.sqrt(dx**2 + dy**2)
            else:
                grad_mag = None
                
            # Flatten CNN features for fast GPU matrix multiplication pooling
            feat_map_flat = feat_map[b].flatten(1) # [C_feat, H*W]
            
            # Transfer labels components map to GPU once per tile to avoid inner loop CPU-to-GPU transfers
            labels_t = torch.from_numpy(labels).to(device)
            
            f_k_list = []
            building_info = []
            
            for area_px, i in valid_comps:
                comp_mask = labels == i
                
                # 1. Geometry Features (scaled to range 0-10)
                area_m2 = area_px * 0.25
                w_box = stats[i, cv2.CC_STAT_WIDTH]
                h_box = stats[i, cv2.CC_STAT_HEIGHT]
                aspect_ratio = min(w_box, h_box) / max(w_box, h_box)
                
                contours, _ = cv2.findContours(comp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                perimeter_px = sum(cv2.arcLength(c, True) for c in contours)
                perimeter_m = perimeter_px * 0.5
                if perimeter_m == 0: perimeter_m = 1.0
                compactness = 4 * np.pi * area_m2 / (perimeter_m**2)
                
                feat_geom = [
                    area_px / 500.0,
                    area_m2 / 125.0,
                    (w_box * 0.5) / 20.0,
                    (h_box * 0.5) / 20.0,
                    aspect_ratio,
                    perimeter_m / 50.0,
                    compactness
                ]
                
                # 2. Depth Features (CPU only - already normalized dynamically)
                if depth_np is not None:
                    comp_d = depth_np[comp_mask]
                    d_mean = comp_d.mean()
                    d_std = comp_d.std()
                    
                    # Single-pass percentile calculation is 5x faster
                    percentiles = np.percentile(comp_d, [10, 50, 90, 95, 99])
                    d_p10, d_med, d_p90, d_p95, d_p99 = percentiles
                    d_range = d_p99 - d_p10
                    d_grad = grad_mag[comp_mask].mean() if grad_mag is not None else 0.0
                    
                    eroded = cv2.erode(comp_mask.astype(np.uint8), np.ones((3,3), np.uint8)) > 0
                    boundary = comp_mask & ~eroded
                    if eroded.sum() > 0 and boundary.sum() > 0:
                        center_edge_diff = depth_np[eroded].mean() - depth_np[boundary].mean()
                    else:
                        center_edge_diff = 0.0
                else:
                    d_mean = d_med = d_std = d_p90 = d_p95 = d_p99 = d_range = d_grad = center_edge_diff = 0.0
                    
                feat_depth = [d_mean, d_med, d_std, d_p90, d_p95, d_p99, d_range, d_grad, center_edge_diff]
                
                # 3. Context Features (scaled to range 0-10)
                feat_context = [
                    tile_density,
                    tile_avg_building_area / 125.0,
                    tile_n_buildings / 10.0
                ]
                
                # 4. CNN Features (GPU matmul pooling: fully differentiable, zero latency, zero transfers)
                comp_mask_t = labels_t == i
                mask_flat = comp_mask_t.flatten().float()
                cnn_feat = torch.matmul(feat_map_flat, mask_flat) / (mask_flat.sum() + 1e-6)
                
                # Concatenate all features and cast to float32
                f_k_np = np.array(feat_geom + feat_depth + feat_context, dtype=np.float32)
                f_k_t = torch.from_numpy(f_k_np).to(device)
                f_k = torch.cat([f_k_t, cnn_feat], dim=0) # [19 + C_feat]
                
                f_k_list.append(f_k)
                
                # Minimum dimension of bounding box in meters
                min_size_m = min(w_box, h_box) * 0.5
                
                building_info.append({
                    'comp_mask': comp_mask,
                    'gt_np': gt_np,
                    'min_size_m': min_size_m,
                    'd_range': d_range if depth_np is not None else 0.0
                })
                
        # Batch evaluation of Gating and Expert heads to optimize convergence and execution speed
        if len(f_k_list) == 0:
            return mask_logits, [], [], [], []
            
        f_k_batch = torch.stack(f_k_list, dim=0) # [num_buildings, in_dim]
        
        # Extrapolating base anchor parameters for High-Rise Expert E3 (fully differentiable)
        min_size_m_t = torch.tensor([b['min_size_m'] for b in building_info], dtype=torch.float32).to(device)
        d_range_t = torch.tensor([b['d_range'] for b in building_info], dtype=torch.float32).to(device)
        
        base3 = F.softplus(self.alpha) * d_range_t + F.softplus(self.beta) * min_size_m_t + 2.0
        
        # Forward pass on all heads
        gate_logits = self.gate(f_k_batch) # [num_buildings, 3]
        w = F.softmax(gate_logits, dim=-1) # [num_buildings, 3]
        
        r1 = self.expert_1(f_k_batch).squeeze(-1) # [num_buildings]
        r2 = self.expert_2(f_k_batch).squeeze(-1) # [num_buildings]
        r3 = self.expert_3(f_k_batch).squeeze(-1) # [num_buildings]
        
        # Numerical safety clamps on log-residuals
        r1 = torch.clamp(r1, -3.0, 3.0)
        r2 = torch.clamp(r2, -3.0, 3.0)
        r3 = torch.clamp(r3, -3.0, 3.0)
        
        # Predictions of individual experts
        H1 = 5.0 * torch.exp(r1)
        H2 = 20.0 * torch.exp(r2)
        H3 = base3 * torch.exp(r3)
        
        # Gated continuous prediction
        pred_heights = w[:, 0] * H1 + w[:, 1] * H2 + w[:, 2] * H3
        
        for k in range(len(f_k_list)):
            info = building_info[k]
            comp_mask = info['comp_mask']
            gt_np = info['gt_np']
            
            pred_height = pred_heights[k]
            gate_logit = gate_logits[k]
            building_preds.append((pred_height, gate_logit, H1[k], H2[k], H3[k]))
            
            if gt_np is not None:
                comp_gt = gt_np[comp_mask]
                true_height = float(np.percentile(comp_gt, 95)) if len(comp_gt) > 0 else 0.0
                building_targets.append(true_height)
                
                # 3-Regime gating label for auxiliary supervision: LOW (<15m), MID (15-30m), HIGH (>=30m)
                true_regime = np.digitize(true_height, [15.0, 30.0])
                true_regime = int(np.clip(true_regime, 0, 2))
                building_regimes.append(true_regime)
                
                # 5-Bin target weight for moderate height-aware sampling/weighting:
                true_bin = np.digitize(true_height, [10.0, 20.0, 30.0, 40.0])
                true_bin = int(np.clip(true_bin, 0, 4))
                building_weights.append(float(BIN_WEIGHTS[true_bin]))
                
        return mask_logits, building_preds, building_targets, building_regimes, building_weights

class BuildingConditionedEstimator(HeightEstimator):
    name = "building_conditioned_height_estimator"
    metric = True

    def __init__(self, cfg_train, nodata: float | None = None, seed: int = 0, device: str | None = None):
        self.cfg = cfg_train
        self.nodata = nodata
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.model = BuildingConditionedHeightNet(w=cfg_train.width, C_feat=16).to(self.device)
        self.d_mean = 0.0
        self.d_std = 1.0

    def _prep_x(self, s: Sample, res: int):
        import cv2
        rgb = np.asarray(s["rgb"], dtype=np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        depth = np.asarray(s["depth"], dtype=np.float32)
        rgb = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (res, res), interpolation=cv2.INTER_LINEAR)
        depth_norm = (depth - self.d_mean) / (self.d_std + 1e-6)
        x = np.concatenate([rgb.transpose(2, 0, 1), depth_norm[None]], axis=0) # 4xHxW
        return x

    def fit(self, train_samples: list[Sample]) -> "BuildingConditionedEstimator":
        # Global depth statistics calculation
        ds = []
        for s in train_samples[: min(len(train_samples), 64)]:
            ds.append(np.asarray(s["depth"], dtype=np.float32).ravel()[::37])
        allc = np.concatenate(ds)
        self.d_mean, self.d_std = float(np.mean(allc)), float(np.std(allc) + 1e-6)
        self.model.d_mean.fill_(self.d_mean)
        self.model.d_std.fill_(self.d_std)

        res = self.cfg.train_res
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        
        rng = np.random.default_rng(self.seed)
        bs = self.cfg.batch_size
        
        self.model.train()
        for epoch in range(self.cfg.epochs):
            order = rng.permutation(len(train_samples))
            ep_loss = 0.0
            nb = 0
            for i in range(0, len(order), bs):
                idx = order[i : i + bs]
                if idx.size < 2:
                    continue
                    
                xs, ys, ms, ds_raw = [], [], [], []
                for j in idx:
                    s = train_samples[j]
                    xs.append(self._prep_x(s, res))
                    
                    gt = np.asarray(s["gt"], dtype=np.float32)
                    valid = np.isfinite(gt)
                    if self.nodata is not None:
                        valid &= gt != self.nodata
                    gt_f = np.where(valid, gt, 0.0)
                    gt_r = cv2.resize(gt_f, (res, res), interpolation=cv2.INTER_LINEAR)
                    valid_r = cv2.resize(valid.astype(np.float32), (res, res), interpolation=cv2.INTER_NEAREST) > 0.5
                    
                    ys.append(gt_r)
                    ms.append(valid_r)
                    
                    depth = np.asarray(s["depth"], dtype=np.float32)
                    depth_r = cv2.resize(depth, (res, res), interpolation=cv2.INTER_LINEAR)
                    ds_raw.append(depth_r)
                    
                x = torch.from_numpy(np.stack(xs)).float().to(self.device)
                y = torch.from_numpy(np.stack(ys)).float().to(self.device)
                m = torch.from_numpy(np.stack(ms)).bool().to(self.device)
                raw_d = torch.from_numpy(np.stack(ds_raw)).float().to(self.device)
                
                opt.zero_grad(set_to_none=True)
                
                # Forward Pass
                mask_logits, preds, targets, regimes, weights = self.model(x, raw_d, y, device=self.device)
                
                # 1. Footprint branch loss: BCE on valid mask
                gt_footprint = (y > 2.0).float()
                loss_footprint = F.binary_cross_entropy_with_logits(mask_logits, gt_footprint, reduction='none')
                loss_footprint = loss_footprint[m].mean()
                
                # 2. Object level losses
                loss_regime = torch.tensor(0.0).to(self.device)
                loss_height = torch.tensor(0.0).to(self.device)
                
                if len(preds) > 0:
                    regime_losses = []
                    height_losses = []
                    for k in range(len(preds)):
                        pred_h, pred_regime_logit = preds[k]
                        target_h = targets[k]
                        target_regime = regimes[k]
                        weight = weights[k]
                        
                        # Cross entropy for regime
                        ce = F.cross_entropy(pred_regime_logit.unsqueeze(0), torch.tensor([target_regime]).to(self.device))
                        regime_losses.append(ce * weight)
                        
                        # Smooth L1 for height scale
                        sl1 = F.smooth_l1_loss(pred_h, torch.tensor([target_h], dtype=torch.float32).to(self.device))
                        height_losses.append(sl1 * weight)
                        
                    loss_regime = torch.stack(regime_losses).mean()
                    loss_height = torch.stack(height_losses).mean()
                    
                total_loss = loss_footprint + 0.5 * loss_regime + 0.1 * loss_height
                
                total_loss.backward()
                opt.step()
                
                ep_loss += float(total_loss.detach())
                nb += 1
                
            print(f"[BuildingConditioned] epoch {epoch+1}/{self.cfg.epochs} loss={ep_loss/max(nb,1):.4f}")
            
        return self

    @torch.no_grad()
    def predict(self, sample: Sample) -> np.ndarray:
        self.model.eval()
        res = self.cfg.train_res
        x = self._prep_x(sample, res)
        xt = torch.from_numpy(x[None]).float().to(self.device)
        
        depth = np.asarray(sample["depth"], dtype=np.float32)
        depth_r = cv2.resize(depth, (res, res), interpolation=cv2.INTER_LINEAR)
        raw_d = torch.from_numpy(depth_r[None]).float().to(self.device)
        
        # Forward pass without targets
        mask_logits, preds, _, _, _ = self.model(xt, raw_d, device=self.device)
        
        # Construct dense predictions
        probs = torch.sigmoid(mask_logits).squeeze(0).cpu().numpy()
        pred_mask_256 = probs > 0.5
        pred_mask = cv2.resize(pred_mask_256.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0.5
        
        # Component-wise dense height reconstruction with roof topology mapping
        pred_h_dense = np.zeros((512, 512), dtype=np.float32)
        
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask.astype(np.uint8), connectivity=8)
        
        valid_comps = []
        for i in range(1, n):
            area_px = stats[i, cv2.CC_STAT_AREA]
            if area_px >= 16:
                valid_comps.append((area_px, i))
        valid_comps.sort(key=lambda x: x[0], reverse=True)
        valid_comps = valid_comps[:25]
        
        # Resize raw depth map to 512x512
        depth_512 = cv2.resize(depth, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        idx = 0
        for area_px, i in valid_comps:
            # If we have a predicted scale for this building component
            if idx < len(preds):
                pred_scale = float(preds[idx][0].item())
                comp_mask = labels == i
                
                # Preserve intra-building roof topology via local normalized relative depth
                comp_depth = depth_512[comp_mask]
                d_min = comp_depth.min()
                d_max = comp_depth.max()
                
                if d_max > d_min:
                    normalized_d = (comp_depth - d_min) / (d_max - d_min)
                else:
                    normalized_d = np.ones_like(comp_depth)
                    
                # Reconstruct height: h_pixel = (S_i - 2.0) * normalized_d + 2.0
                comp_h = (pred_scale - 2.0) * normalized_d + 2.0
                comp_h = np.maximum(comp_h, 2.0)
                pred_h_dense[comp_mask] = comp_h
                
            idx += 1
            
        h, w = np.asarray(sample["gt"]).shape[:2]
        return cv2.resize(pred_h_dense, (w, h), interpolation=cv2.INTER_LINEAR)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))
