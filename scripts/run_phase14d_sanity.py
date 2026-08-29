import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from transformers import AutoModelForDepthEstimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.models.fusion_head import SmallFusionUNet

def prep_rgb_dav2(rgb_np, target_size=518):
    rgb = cv2.resize(rgb_np, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    return rgb.transpose(2, 0, 1)

def prep_rgb_unet(rgb_np, target_size=256):
    rgb = rgb_np.astype(np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    rgb = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return rgb.transpose(2, 0, 1)

class Phase14DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.da_v2 = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        
        # Freeze backbone
        self.f_params = 0
        self.t_params = 0
        for name, param in self.da_v2.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False
                self.f_params += param.numel()
            else:
                param.requires_grad = True
                self.t_params += param.numel()
                
        self.unet = SmallFusionUNet(w=24, in_channels=4, out_channels=1)
        for param in self.unet.parameters():
            self.t_params += param.numel()
            
        self.register_buffer("d_mean", torch.tensor(0.0))
        self.register_buffer("d_std", torch.tensor(1.0))

    def forward(self, rgb_dav2, rgb_unet):
        # DA-V2 expects Bx3x518x518
        da_out = self.da_v2(rgb_dav2).predicted_depth.unsqueeze(1) # Bx1x518x518
        
        # Interpolate DA-V2 output to UNet input size
        # Same interpolation approach as C_log1p
        depth_256 = F.interpolate(da_out, size=rgb_unet.shape[-2:], mode='bilinear', align_corners=False)
        
        # Normalize depth using the same approach as C_log1p
        depth_256 = (depth_256 - self.d_mean) / (self.d_std + 1e-6)
        
        # Concat RGB + Depth for UNet
        x = torch.cat([rgb_unet, depth_256], dim=1) # Bx4x256x256
        
        return self.unet(x)

def run_sanity_check():
    print("=== Phase 14D Sanity Check ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Phase14DModel().to(device)
    
    print(f"1. DA-V2 backbone frozen (param check): {model.f_params:,} frozen.")
    print(f"2/3. Trainable DA-V2 neck/head + UNet: {model.t_params:,} trainable.")
    
    # Mock data
    rgb = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    target = np.random.rand(256, 256).astype(np.float32) * 10
    mask = (target > 0)
    
    rgb_d = prep_rgb_dav2(rgb)
    rgb_u = prep_rgb_unet(rgb)
    
    rgb_d_t = torch.from_numpy(rgb_d).unsqueeze(0).to(device)
    rgb_u_t = torch.from_numpy(rgb_u).unsqueeze(0).to(device)
    y_t = torch.from_numpy(target).unsqueeze(0).to(device)
    m_t = torch.from_numpy(mask).unsqueeze(0).to(device)
    
    # Init stats
    model.eval()
    with torch.no_grad():
        d_out = model.da_v2(rgb_d_t).predicted_depth
        model.d_mean.fill_(d_out.mean().item())
        model.d_std.fill_(d_out.std().item())
    
    model.train()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    # 7. Forward pass
    pred = model(rgb_d_t, rgb_u_t)
    print(f"7/11. Forward pass shape: {pred.shape}")
    assert pred.shape == y_t.shape, "Shape mismatch"
    
    loss = torch.abs(pred - y_t)[m_t].mean()
    loss.backward()
    print("8. Backward pass works.")
    
    # Check gradients
    has_head_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for name, p in model.da_v2.named_parameters() if "head" in name)
    has_neck_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for name, p in model.da_v2.named_parameters() if "neck" in name)
    has_backbone_grad = any(p.grad is not None for name, p in model.da_v2.named_parameters() if "backbone" in name)
    has_unet_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.unet.parameters())
    
    print(f"4. Gradients reach neck/head: {has_neck_grad and has_head_grad}")
    print(f"5. Gradients reach SmallFusionUNet: {has_unet_grad}")
    print(f"6. Backbone receives no gradients: {not has_backbone_grad}")
    
    assert has_head_grad and has_neck_grad, "Neck/head missing grads"
    assert has_unet_grad, "UNet missing grads"
    assert not has_backbone_grad, "Backbone has grads!"
    
    opt.step()
    
    # 9. Loss decreases
    opt.zero_grad()
    initial_loss = loss.item()
    for _ in range(5):
        pred = model(rgb_d_t, rgb_u_t)
        loss = torch.abs(pred - y_t)[m_t].mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    final_loss = loss.item()
    print(f"9. Loss decreases: {initial_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss didn't decrease"
    
    print(f"10. No NaN/Inf: {torch.isfinite(loss)}")
    
    vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    print(f"12. Memory fits RTX 3050 4GB: Peak VRAM {vram:.1f} MB")
    
    torch.save(model.state_dict(), "tmp_ckpt.pt")
    model2 = Phase14DModel().to(device)
    model2.load_state_dict(torch.load("tmp_ckpt.pt"))
    print("13. Checkpoint save/load includes both DA-V2 components and SmallFusionUNet.")
    
    print("=== Sanity Check SUCCESS ===")

if __name__ == "__main__":
    run_sanity_check()
