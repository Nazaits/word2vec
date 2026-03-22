import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from word2vec import SGNS, CBOW

# ── Load training results ──
with open("training_results.json", "r") as f:
    results = json.load(f)

# ── Plot 1: CPU vs GPU loss curves + Dimension comparison ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

if "cpu_100d_5ep" in results and "gpu_100d_5ep" in results:
    losses_cpu = results["cpu_100d_5ep"]["losses"]
    losses_gpu = results["gpu_100d_5ep"]["losses"]
    axes[0].plot(range(len(losses_cpu)), losses_cpu, marker='o', label=f'CPU ({results["cpu_100d_5ep"]["time"]:.0f}s)')
    axes[0].plot(range(len(losses_gpu)), losses_gpu, marker='s', label=f'GPU ({results["gpu_100d_5ep"]["time"]:.0f}s)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('CPU vs GPU Training (dim=100)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

for dim in [50, 100, 300]:
    key = f"gpu_{dim}d_10ep"
    if key in results:
        losses = results[key]["losses"]
        axes[1].plot(range(len(losses)), losses, marker='o', label=f'dim={dim} ({results[key]["time"]:.0f}s)')

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Loss by Embedding Dimension (GPU)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/training_comparison.png', dpi=150)
plt.close()
print("Saved plots/training_comparison.png")

# ── Plot 2: Training time bar chart ──
fig, ax = plt.subplots(figsize=(8, 5))

labels = []
times = []
for key in ["cpu_100d_5ep", "gpu_100d_5ep"]:
    if key in results:
        labels.append(key.replace("_", " ").upper())
        times.append(results[key]["time"])

if labels:
    ax.bar(labels, times, color=['#4a90d9', '#e74c3c'])
    ax.set_ylabel('Time (seconds)')
    ax.set_title('CPU vs GPU Training Time (dim=100, 5 epochs)')
    for i, t in enumerate(times):
        ax.text(i, t + 5, f'{t:.1f}s', ha='center', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('plots/training_times.png', dpi=150)
    plt.close()
    print("Saved plots/training_times.png")

# ── Plot 3: SGNS vs CBOW loss curves ──
sgns_key = "gpu_300d_10ep"
cbow_key = "cbow_gpu_300d_10ep"

if sgns_key in results and cbow_key in results:
    fig, ax = plt.subplots(figsize=(8, 5))

    losses_sgns = results[sgns_key]["losses"]
    losses_cbow = results[cbow_key]["losses"]

    ax.plot(range(len(losses_sgns)), losses_sgns, marker='o', label=f'SGNS ({results[sgns_key]["time"]:.0f}s)')
    ax.plot(range(len(losses_cbow)), losses_cbow, marker='s', label=f'CBOW ({results[cbow_key]["time"]:.0f}s)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('SGNS vs CBOW Training Loss (dim=300, GPU)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/sgns_vs_cbow.png', dpi=150)
    plt.close()
    print("Saved plots/sgns_vs_cbow.png")

# ── t-SNE: SGNS vs CBOW side by side ──
word_groups = {
    'Royalty': ['king', 'queen', 'prince', 'princess', 'throne', 'crown', 'royal'],
    'Countries': ['france', 'germany', 'italy', 'spain', 'england', 'japan', 'china'],
    'Animals': ['dog', 'cat', 'horse', 'bird', 'fish', 'lion', 'tiger'],
    'Science': ['physics', 'chemistry', 'biology', 'mathematics', 'science', 'theory'],
    'Numbers': ['one', 'two', 'three', 'four', 'five', 'six', 'seven'],
}
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

models_to_plot = []

# Load SGNS 300d
model_sgns = SGNS(embedding_dim=300)
try:
    model_sgns.load_embeddings("embeddings/embeddings_sgns_300d.npy")
    models_to_plot.append(("SGNS (300d)", model_sgns))
    print("Loaded SGNS 300d embeddings")
except FileNotFoundError:
    model_sgns = SGNS(embedding_dim=100)
    try:
        model_sgns.load_embeddings("embeddings/embeddings_sgns_100d.npy")
        models_to_plot.append(("SGNS (100d)", model_sgns))
        print("Loaded SGNS 100d embeddings")
    except FileNotFoundError:
        print("No SGNS embeddings found")

# Load CBOW 300d
model_cbow = CBOW(embedding_dim=300)
try:
    model_cbow.load_embeddings("embeddings/embeddings_cbow_300d.npy")
    models_to_plot.append(("CBOW (300d)", model_cbow))
    print("Loaded CBOW 300d embeddings")
except FileNotFoundError:
    print("No CBOW embeddings found")

ncols = len(models_to_plot)
if ncols > 0:
    fig, axes = plt.subplots(1, ncols, figsize=(14 * ncols // 2, 10))
    if ncols == 1:
        axes = [axes]

    for ax, (model_name, model) in zip(axes, models_to_plot):
        all_words = []
        all_embeddings = []
        word_to_group = {}

        for group_name, words in word_groups.items():
            for word in words:
                if word in model.word2id:
                    all_words.append(word)
                    all_embeddings.append(model.E[model.word2id[word]])
                    word_to_group[word] = group_name

        all_embeddings = np.array(all_embeddings)

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, len(all_words) - 1))
        embeddings_2d = tsne.fit_transform(all_embeddings)

        group_names = list(word_groups.keys())
        for i, group_name in enumerate(group_names):
            mask = [word_to_group[w] == group_name for w in all_words]
            points = embeddings_2d[mask]
            ax.scatter(points[:, 0], points[:, 1], c=colors[i], label=group_name, s=100, alpha=0.7, edgecolors='white', linewidth=0.5)

        for i, word in enumerate(all_words):
            ax.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        fontsize=9, ha='center', va='bottom',
                        textcoords="offset points", xytext=(0, 6))

        ax.set_title(f'{model_name} - Word Clusters (t-SNE)', fontsize=14)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('t-SNE dimension 1')
        ax.set_ylabel('t-SNE dimension 2')

    plt.tight_layout()
    plt.savefig('plots/word_clusters.png', dpi=150)
    plt.close()
    print("Saved plots/word_clusters.png")

# ── Plot 4: Similarity & Analogy comparison tables as images ──
test_words = ['king', 'computer', 'france', 'dog']
analogies = [
    ('man', 'woman', 'king'),
    ('france', 'paris', 'germany'),
    ('big', 'bigger', 'small'),
]

# SGNS vs CBOW comparison table
if len(models_to_plot) > 0:
    fig, axes = plt.subplots(len(models_to_plot), 1, figsize=(16, 5 * len(models_to_plot)))
    if len(models_to_plot) == 1:
        axes = [axes]

    for ax, (model_name, model) in zip(axes, models_to_plot):
        ax.axis('off')

        rows = []
        # Similar words
        for word in test_words:
            similar = model.find_similar_words(word, n=5)
            similar = [str(s) for s in similar]
            rows.append([f"Similar to '{word}'"] + similar)

        # Analogies
        for w1, w2, w3 in analogies:
            result = model.find_analogy(w1, w2, w3)
            result = [str(r) for r in result]
            rows.append([f"{w1} -> {w2} :: {w3} -> ?"] + result)

        col_labels = ['Query', '#1', '#2', '#3', '#4', '#5']
        table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)

        # Style header
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#2c3e50')
            table[0, j].set_text_props(color='white', fontweight='bold')

        # Alternate row colors
        for i in range(len(rows)):
            color = '#f7f9fc' if i % 2 == 0 else '#ffffff'
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor(color)

        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('plots/similarity_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plots/similarity_comparison.png")

# ── Plot 5: Dimension quality comparison table ──
dim_rows = {}
for dim in [50, 100, 300]:
    model_dim = SGNS(embedding_dim=dim)
    try:
        model_dim.load_embeddings(f"embeddings/embeddings_sgns_{dim}d.npy")
    except FileNotFoundError:
        continue

    rows = []
    for word in test_words:
        similar = model_dim.find_similar_words(word, n=5)
        similar = [str(s) for s in similar]
        rows.append([f"Similar to '{word}'"] + similar)

    for w1, w2, w3 in analogies:
        result = model_dim.find_analogy(w1, w2, w3)
        result = [str(r) for r in result]
        rows.append([f"{w1} -> {w2} :: {w3} -> ?"] + result)

    dim_rows[dim] = rows

if dim_rows:
    dims = list(dim_rows.keys())
    fig, axes = plt.subplots(len(dims), 1, figsize=(16, 5 * len(dims)))
    if len(dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        ax.axis('off')
        rows = dim_rows[dim]
        col_labels = ['Query', '#1', '#2', '#3', '#4', '#5']
        table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)

        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#2c3e50')
            table[0, j].set_text_props(color='white', fontweight='bold')

        for i in range(len(rows)):
            color = '#f7f9fc' if i % 2 == 0 else '#ffffff'
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor(color)

        ax.set_title(f'SGNS dim={dim}', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('plots/dimension_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plots/dimension_comparison.png")

# ── Interactive word explorer ──
print("\n" + "=" * 50)
print("Interactive Word Explorer")
print("Type words separated by commas to plot them.")
print("Type 'quit' to exit.")
print("=" * 50)

if models_to_plot:
    while True:
        user_input = input("\nEnter words (comma-separated): ").strip()
        if user_input.lower() == 'quit':
            break

        words = [w.strip().lower() for w in user_input.split(',')]

        # Check validity against all models
        all_valid = set()
        for _, model in models_to_plot:
            all_valid.update(w for w in words if w in model.word2id)
        invalid_words = [w for w in words if w not in all_valid]

        if invalid_words:
            print(f"Words not in vocabulary: {invalid_words}")

        valid_words = [w for w in words if w in all_valid]

        if len(valid_words) < 2:
            print("Need at least 2 valid words to plot.")
            continue

        ncols = len(models_to_plot)
        fig, axes = plt.subplots(1, ncols, figsize=(10 * ncols, 8))
        if ncols == 1:
            axes = [axes]

        for ax, (model_name, model) in zip(axes, models_to_plot):
            model_valid = [w for w in valid_words if w in model.word2id]
            if len(model_valid) < 2:
                ax.set_title(f'{model_name} - Not enough valid words')
                continue

            embeddings = np.array([model.E[model.word2id[w]] for w in model_valid])

            tsne_interactive = TSNE(n_components=2, random_state=42, perplexity=min(5, len(model_valid) - 1))
            coords = tsne_interactive.fit_transform(embeddings)

            ax.scatter(coords[:, 0], coords[:, 1], c='#3498db', s=120, alpha=0.7, edgecolors='white', linewidth=0.5)

            for i, word in enumerate(model_valid):
                ax.annotate(word, (coords[i, 0], coords[i, 1]),
                            fontsize=11, ha='center', va='bottom',
                            textcoords="offset points", xytext=(0, 8))

            ax.set_title(f'{model_name} - Word Visualization (t-SNE)')
            ax.grid(True, alpha=0.2)

        filename = 'plots/custom_words.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved {filename}")
