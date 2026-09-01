content = open('scripts/run_phase43_augmented_unet.py', encoding='utf-8').read()

content = content.replace(
    "train_samples = load_samples(train_ids, max_samples=128)",
    "train_samples = load_samples(train_ids, max_samples=MAX_TRAIN, label='train')"
)
content = content.replace(
    "val_samples   = load_samples(val_ids,   max_samples=64)",
    "val_samples   = load_samples(val_ids,   max_samples=MAX_VAL,   label='val')"
)
content = content.replace(
    "test_samples  = load_samples(test_ids,  max_samples=32)",
    "test_samples  = load_samples(test_ids,  max_samples=MAX_TEST,  label='test')"
)

# Add flush=True to epoch prints
old = "print(f\"    epoch {epoch+1}/{epochs}  loss={ep_loss/max(nb,1):.4f}  val_iou={val_iou:.4f}\")"
new = "print(f\"    epoch {epoch+1}/{epochs}  loss={ep_loss/max(nb,1):.4f}  val_iou={val_iou:.4f}\", flush=True)"
content = content.replace(old, new)

old2 = "print(f\"  [Train] Config={config_mode} Seed={seed} Epochs={epochs}\")"
new2 = "print(f\"  [Train] Config={config_mode} Seed={seed} Epochs={epochs}\", flush=True)"
content = content.replace(old2, new2)

open('scripts/run_phase43_augmented_unet.py', 'w', encoding='utf-8').write(content)
print('Patched successfully.')
