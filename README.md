# 🗜️ MNIST Quantization From Scratch

A beginner-friendly project that teaches **AI model quantization** by doing it hands-on — training a neural network, compressing it, and measuring exactly what changes.

## What is Quantization?

Neural networks store their "knowledge" as millions of decimal numbers called **weights** (e.g. `0.38291748192`). These numbers are precise but take up a lot of memory.

**Quantization** means rounding those numbers to use fewer bits of storage:

| Format  | Bits | Example value stored |
|---------|------|----------------------|
| float64 | 64   | `0.38291748192...`   |
| float32 | 32   | `0.3829175`          |
| float16 | 16   | `0.3828`             |
| int8    | 8    | `49` (scaled integer)|

Fewer bits = **smaller file** + **faster inference**. The trade-off: a small accuracy loss.

This is how tools like [llama.cpp](https://github.com/ggerganov/llama.cpp) let you run billion-parameter AI models on a laptop.

---

## Project Structure

```
mnist-quantization/
├── train.py       # Train a neural network on handwritten digits
├── quantize.py    # Compress it into float32, float16, and int8
├── compare.py     # Benchmark size, accuracy, and speed
├── requirements.txt
└── results/
    └── results.md # Auto-generated comparison table
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python train.py

# 3. Quantize it
python quantize.py

# 4. Compare results
python compare.py
```

---

## Results

<!-- After running compare.py, paste results/results.md here -->

![Results](results.png)

---

## Interesting Finding: int8 Is Tricky!

You'll notice int8 accuracy drops dramatically. Why? Because `int8` only stores integers from -128 to 127, but our weights are tiny decimals like `0.003`. When cast directly to int8, they all round to **0**, destroying the model.

Real-world quantization tools (like llama.cpp) handle this by **scaling** the weights before converting — e.g., multiplying everything by 1000 first, then dividing back later at inference time. This is called **zero-point quantization** and is a great next step to explore!

---

## What I Learned

- Neural network weights are just arrays of numbers
- Changing the **data type** (float64 → int8) is the core of quantization
- Smaller models run faster and use less memory
- There's a size vs. accuracy trade-off — but it's surprisingly small!

---

## Tech Stack

- **Python** — scikit-learn, numpy, joblib
- **Dataset** — [MNIST](http://yann.lecun.com/exdb/mnist/) (70,000 handwritten digit images)
- **Model** — Multi-layer Perceptron (MLP) classifier

## Web App

Run an interactive digit recognizer in your browser:

```bash
pip install flask
python app.py
```

Then open `http://localhost:5000`, draw a digit, and see how each model version predicts it!

## Limitations & Future Improvements

- **7, 5, and 8 are hard to distinguish** — at 8x8 pixels, these digits look very similar after compression
- **Small training set** — only 1,797 examples vs. 70,000 in the full MNIST dataset
- **Handwriting mismatch** — the model learned one style of handwriting, yours may differ

### Ideas for improvement
- Retrain on the full MNIST dataset for better accuracy
- Better preprocess drawings before predicting (center and scale the digit)
- Add more hidden layers to the neural network
