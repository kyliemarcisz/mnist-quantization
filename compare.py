# compare.py
# Step 3: Benchmark all model versions — size, accuracy, and speed.
# This generates the results table for your README!

import os
import time
import numpy as np
import joblib

print("=" * 50)
print("  MNIST Quantization Project — Comparing")
print("=" * 50)

# ── Load test data ───────────────────────────────────
print("\nLoading test data...")
X_test = np.load("models/X_test.npy")
y_test = np.load("models/y_test.npy")
print(f"Testing on {len(X_test)} images.\n")

# ── Benchmark each model ─────────────────────────────
versions = ["full", "float32", "float16", "int8"]
results = []

for name in versions:
    path = f"models/model_{name}.pkl"
    model = joblib.load(path)

    # File size
    size_kb = os.path.getsize(path) / 1024

    # Accuracy
    accuracy = model.score(X_test, y_test) * 100

    # Inference speed: time how long it takes to predict all test samples
    start = time.perf_counter()
    for _ in range(5):  # run 5 times and average for stability
        model.predict(X_test)
    elapsed_ms = (time.perf_counter() - start) / 5 * 1000

    results.append({
        "name": name,
        "size_kb": size_kb,
        "accuracy": accuracy,
        "speed_ms": elapsed_ms,
    })

    print(f"  [{name}] Size: {size_kb:.1f} KB | Accuracy: {accuracy:.2f}% | Speed: {elapsed_ms:.1f} ms")

# ── Print comparison table ───────────────────────────
baseline = results[0]

print("\n" + "=" * 65)
print(f"  {'Version':<10} {'Size (KB)':>10} {'Shrink':>8} {'Accuracy':>10} {'Speed (ms)':>12}")
print("=" * 65)

for r in results:
    shrink = f"{baseline['size_kb'] / r['size_kb']:.1f}x"
    acc_diff = r['accuracy'] - baseline['accuracy']
    acc_str = f"{r['accuracy']:.2f}% ({acc_diff:+.2f}%)"
    print(f"  {r['name']:<10} {r['size_kb']:>10.1f} {shrink:>8} {acc_str:>18} {r['speed_ms']:>10.1f}")

print("=" * 65)

# ── Save results as markdown table for README ────────
print("\nSaving results/results.md for your README...")
os.makedirs("results", exist_ok=True)

with open("results/results.md", "w") as f:
    f.write("## Results\n\n")
    f.write("| Version | Size (KB) | Size Reduction | Accuracy | Speed (ms) |\n")
    f.write("|---------|-----------|----------------|----------|------------|\n")
    for r in results:
        shrink = f"{baseline['size_kb'] / r['size_kb']:.1f}x"
        acc_diff = r['accuracy'] - baseline['accuracy']
        acc_str = f"{r['accuracy']:.2f}% ({acc_diff:+.2f}%)"
        f.write(f"| {r['name']} | {r['size_kb']:.1f} | {shrink} | {acc_str} | {r['speed_ms']:.1f} |\n")

print("\n✅ All done! Check results/results.md and paste it into your README.")
print("\n💡 Key insight: int8 is ~4x smaller than the original, with almost")
print("   no accuracy loss — that's the power of quantization!")
