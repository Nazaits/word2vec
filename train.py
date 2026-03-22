import time
import json
import numpy as np
from word2vec import SGNS, CBOW

results = {}

# ── CPU vs GPU comparison (dim=100, 5 epochs) ──
print("=" * 50)
print("CPU vs GPU comparison (dim=100, 5 epochs)")
print("=" * 50)

# CPU training
model_cpu = SGNS(embedding_dim=100)
st = time.time()
losses_cpu = model_cpu.train(epochs=5, gpu=False)
cpu_time = time.time() - st
print(f"CPU Training time: {cpu_time:.2f} seconds")

results["cpu_100d_5ep"] = {
    "time": cpu_time,
    "losses": [float(l) for l in losses_cpu],
}

# GPU training
model_gpu = SGNS(embedding_dim=100)
st = time.time()
losses_gpu = model_gpu.train(epochs=5, gpu=True)
gpu_time = time.time() - st
print(f"GPU Training time: {gpu_time:.2f} seconds")

results["gpu_100d_5ep"] = {
    "time": gpu_time,
    "losses": [float(l) for l in losses_gpu],
}

# ── Dimension comparison on GPU (50, 100, 300 dims, 10 epochs) ──
print("\n" + "=" * 50)
print("Dimension comparison on GPU (10 epochs)")
print("=" * 50)

for dim in [50, 100, 300]:
    print(f"\n--- dim={dim} ---")
    model = SGNS(embedding_dim=dim)
    st = time.time()
    losses = model.train(epochs=10, gpu=True)
    train_time = time.time() - st
    print(f"dim={dim} Training time: {train_time:.2f} seconds")

    results[f"gpu_{dim}d_10ep"] = {
        "time": train_time,
        "losses": [float(l) for l in losses],
    }

    model.save_embeddings(f"embeddings/embeddings_sgns_{dim}d.npy")

# ── CBOW training (dim=300, 10 epochs) ──
print("\n" + "=" * 50)
print("CBOW training (dim=300, 10 epochs)")
print("=" * 50)

model_cbow = CBOW(embedding_dim=300)
st = time.time()
losses_cbow = model_cbow.train(epochs=10, gpu=True)
cbow_time = time.time() - st
print(f"CBOW Training time: {cbow_time:.2f} seconds")

results["cbow_gpu_300d_10ep"] = {
    "time": cbow_time,
    "losses": [float(l) for l in losses_cbow],
}

model_cbow.save_embeddings("embeddings/embeddings_cbow_300d.npy")

# Save all results
with open("training_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll results saved to training_results.json")
