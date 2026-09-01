import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs' / 'phase51_corrected_unet'
CHECKPOINTS = OUT / 'checkpoints'
MANIFEST = OUT / 'RUN_MANIFEST.json'

# All eight runs in the completed matrix executed with batch 16.
BATCH_SIZE = 16

for checkpoint in CHECKPOINTS.glob('*_best.pt'):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    metadata = payload.setdefault('metadata', {})
    metadata['batch_size'] = BATCH_SIZE
    metadata['batch_size_execution_note'] = 'Batch 16 passed stability, but benchmark was slower than batch 8; this completed matrix nevertheless used batch 16 throughout.'
    torch.save(payload, checkpoint)

manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
for entry in manifest.values():
    entry['batch_size'] = BATCH_SIZE
    entry['batch_size_execution_note'] = 'Completed matrix used batch 16; batch 8 was benchmarked faster but was not applied before completion.'
MANIFEST.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('UPDATED_CHECKPOINTS', len(list(CHECKPOINTS.glob('*_best.pt'))))
print('RECORDED_BATCH_SIZE', BATCH_SIZE)
