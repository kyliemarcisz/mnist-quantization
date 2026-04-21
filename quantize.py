# quantize.py
# Step 2: Load the trained model and create quantized versions.
#
# WHAT IS HAPPENING HERE?
# A neural network stores its "knowledge" as millions of decimal numbers
# called weights (e.g. 0.38291748). Quantization means we round those
# numbers to use fewer bits of storage:
#
#   float64 → 64 bits per number  (most precise, largest)
#   float32 → 32 bits per number
#   float16 → 16 bits per number
#   int8    →  8 bits per number  (least precise, smallest)
#
# Fewer bits = smaller file = faster math. The trade-off: a tiny loss in accuracy.

import os
import numpy as np
import joblib

print("=" * 50)
print("  MNIST Quantization Project — Quantizing")
print("=" * 50)

# ── Load the original model ──────────────────────────
print("\n[1/4] Loading full-precision model...")
model = joblib.load("models/model_full.pkl")


def quantize_model(model, dtype):
    """
    Return a copy of the model with weights cast to a new data type.
    This is the core of quantization — we're just changing how precise
    each number is stored.
    """
    import copy
    quantized = copy.deepcopy(model)

    # coefs_ holds the weight matrices between each layer
    quantized.coefs_ = [w.astype(dtype) for w in model.coefs_]
    # intercepts_ holds the bias values for each layer
    quantized.intercepts_ = [b.astype(dtype) for b in model.intercepts_]

    return quantized


def print_weight_sample(model, label):
    """Show a few raw weight values so you can SEE the precision difference."""
    sample = model.coefs_[0][0][:5]  # first 5 weights of first layer
    print(f"      [{label}] sample weights: {sample}")


# ── Inspect original weights ─────────────────────────
print("\n[2/4] Original weight precision (float64):")
print_weight_sample(model, "float64")

# ── Create quantized versions ────────────────────────
print("\n[3/4] Creating quantized versions...")

versions = {
    "float32": np.float32,
    "float16": np.float16,
    "int8":    np.int8,
}

for name, dtype in versions.items():
    q_model = quantize_model(model, dtype)
    path = f"models/model_{name}.pkl"
    joblib.dump(q_model, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"      {name:8s} → saved ({size_kb:.1f} KB)")
    print_weight_sample(q_model, name)

# ── Show size comparison ─────────────────────────────
print("\n[4/4] File size summary:")
all_versions = ["full", "float32", "float16", "int8"]
for v in all_versions:
    path = f"models/model_{v}.pkl"
    size_kb = os.path.getsize(path) / 1024
    bar = "█" * int(size_kb / 20)
    print(f"      {v:8s}: {size_kb:6.1f} KB  {bar}")

print("\n✅ Done! Now run: python compare.py")
