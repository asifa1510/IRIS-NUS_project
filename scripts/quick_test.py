from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ch_regime_dataset.simulator import random_config, simulate_batch
from ch_regime_dataset.validation import validate_dataset

cfg = random_config(trajectory_id=0, split="baseline", seed=42)
df = simulate_batch(cfg)

print("\nHEAD:")
print(df.head())

print("\nTAIL:")
print(df.tail())

print("\nVALIDATION:")
summary = validate_dataset(df)

for k, v in summary.items():
    print(f"{k}: {v}")