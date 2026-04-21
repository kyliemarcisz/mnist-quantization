# train.py
# Step 1: Train a simple model on the MNIST handwritten digits dataset
# and save it so we can quantize it later.

import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("=" * 50)
print("  MNIST Quantization Project — Training")
print("=" * 50)

# ── 1. Load the dataset ──────────────────────────────
print("\n[1/4] Loading digits dataset...")
# We use sklearn's built-in digits dataset — 1,797 images of handwritten
# digits (0–9), each 8x8 pixels. Same idea as MNIST but works offline!
digits = load_digits()
X, y = digits.data, digits.target.astype(str)
print(f"      Loaded {len(X)} images, each with {X.shape[1]} pixels (8x8 grid).")

# ── 2. Preprocess ────────────────────────────────────
print("\n[2/4] Preprocessing (scaling pixel values)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ── 3. Train the model ───────────────────────────────
print("\n[3/4] Training neural network (hidden layers: 128, 64)...")
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=20,
    random_state=42,
    verbose=False,
)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"      Training complete! Test accuracy: {accuracy * 100:.2f}%")

# ── 4. Save everything ───────────────────────────────
print("\n[4/4] Saving model and test data...")
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model_full.pkl")
np.save("models/X_test.npy", X_test)
np.save("models/y_test.npy", y_test)

size_kb = os.path.getsize("models/model_full.pkl") / 1024
print(f"      Saved! Full model size: {size_kb:.1f} KB")
print("\n✅ Done! Now run: python quantize.py")
