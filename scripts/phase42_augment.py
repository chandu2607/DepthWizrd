import cv2
import numpy as np

def augment_sample(sample, config_mode, rng):
    """
    Apply augmentations based on config mode.
    Modes: 'A' (None), 'B' (Geom), 'C' (Geom+Photo), 'D' (Geom+Photo+Multiscale)
    """
    if config_mode == 'A':
        return sample
        
    s = sample.copy()
    rgb = s["rgb"].copy()
    gt = s["gt"].copy()
    depth = s["depth"].copy()
    mask_bldg = s["mask_bldg"].copy().astype(np.uint8) if "mask_bldg" in s else None
    
    h, w = rgb.shape[:2]
    
    # Photometric (C, D)
    if config_mode in ['C', 'D']:
        if rng.random() < 0.5:
            # Brightness/Contrast
            alpha = rng.uniform(0.8, 1.2) # contrast
            beta = rng.uniform(-20, 20)   # brightness
            rgb = cv2.convertScaleAbs(rgb, alpha=alpha, beta=beta)
        if rng.random() < 0.5:
            # Saturation/Hue
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-5, 5)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.85, 1.15), 0, 255)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        if rng.random() < 0.3:
            # Noise
            noise = rng.normal(0, 10, rgb.shape).astype(np.float32)
            rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if rng.random() < 0.2:
            rgb = cv2.GaussianBlur(rgb, (3, 3), 0)
            
    # Geometric (B, C, D)
    if config_mode in ['B', 'C', 'D']:
        if rng.random() < 0.5:
            rgb = cv2.flip(rgb, 1)
            gt = cv2.flip(gt, 1)
            depth = cv2.flip(depth, 1)
            if mask_bldg is not None: mask_bldg = cv2.flip(mask_bldg, 1)
        if rng.random() < 0.5:
            rgb = cv2.flip(rgb, 0)
            gt = cv2.flip(gt, 0)
            depth = cv2.flip(depth, 0)
            if mask_bldg is not None: mask_bldg = cv2.flip(mask_bldg, 0)
            
        rot_choice = rng.choice([0, 90, 180, 270])
        if rot_choice != 0:
            k = rot_choice // 90
            rgb = np.rot90(rgb, k).copy()
            gt = np.rot90(gt, k).copy()
            depth = np.rot90(depth, k).copy()
            if mask_bldg is not None: mask_bldg = np.rot90(mask_bldg, k).copy()
            h, w = rgb.shape[:2]
            
        # Small rotation & scale (Affine)
        if rng.random() < 0.5:
            angle = rng.uniform(-15, 15)
            scale = rng.uniform(0.8, 1.2)
            center = (w/2, h/2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            
            rgb = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
            gt = cv2.warpAffine(gt, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=s["nodata"])
            depth = cv2.warpAffine(depth, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
            if mask_bldg is not None: 
                mask_bldg = cv2.warpAffine(mask_bldg, M, (w, h), flags=cv2.INTER_NEAREST)
                
    # Multi-scale Crop (D)
    if config_mode == 'D':
        if rng.random() < 0.6:
            # Medium or Building crop
            crop_sz = int(rng.uniform(0.3, 0.7) * min(w, h))
            cx = rng.integers(crop_sz//2, w - crop_sz//2)
            cy = rng.integers(crop_sz//2, h - crop_sz//2)
            x1, y1 = cx - crop_sz//2, cy - crop_sz//2
            x2, y2 = x1 + crop_sz, y1 + crop_sz
            
            rgb = rgb[y1:y2, x1:x2]
            gt = gt[y1:y2, x1:x2]
            depth = depth[y1:y2, x1:x2]
            if mask_bldg is not None: mask_bldg = mask_bldg[y1:y2, x1:x2]
            
    s["rgb"] = rgb
    s["gt"] = gt
    s["depth"] = depth
    if mask_bldg is not None: s["mask_bldg"] = mask_bldg
    
    return s
