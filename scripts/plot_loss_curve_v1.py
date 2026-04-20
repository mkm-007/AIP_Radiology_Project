from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "generation_train_history_v1.csv")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
ax.plot(df["epoch"], df["valid_loss"], marker="s", label="Valid Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("Generation Model Training Loss Curve")
ax.legend(); ax.grid(True)
plt.tight_layout()
out = PROJECT_ROOT / "results" / "figures" / "loss_curve_v1.png"
plt.savefig(out, dpi=180); plt.close()
print(f"Saved: {out}")