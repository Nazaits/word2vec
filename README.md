# Word2Vec — Pure NumPy

Skip-gram with negative sampling (SGNS) and CBOW, implemented from scratch using only NumPy. GPU acceleration via CuPy (drop-in replacement). Trained on the [text8](http://mattmahoney.net/dc/textdata.html) dataset.

## Files

| File | Description |
|------|-------------|
| `word2vec.py` | `SGNS` and `CBOW` model classes |
| `preprocessing.py` | Vocabulary building, subsampling, training pair generation |
| `train.py` | Training script — CPU/GPU comparison, dimension comparison, CBOW training |
| `evaluate.py` | Visualization generation and interactive word explorer |
| `requirements.txt` | Python dependencies |

## Structure

```
embeddings/          — saved .npy embedding matrices
plots/               — generated visualizations
resources/           — handwritten gradient derivations
training_results.json — training times and loss curves
text8                — dataset (place in root)
```

## API

### `SGNS(embedding_dim)`

| Method | Description |
|--------|-------------|
| `train(epochs, learning_rate, k, batch_size, gpu)` | Train the model. `k` = number of negative samples, `gpu=True` uses CuPy for GPU acceleration |
| `save_embeddings(file_name)` | Save the embedding matrix E to a `.npy` file |
| `load_embeddings(file_name)` | Load a previously saved embedding matrix |
| `find_similar_words(word, n)` | Return the top `n` most similar words by cosine similarity |
| `find_similar(v_word, n)` | Return the top `n` most similar words to an arbitrary vector |
| `find_analogy(word1, word2, word3)` | Solve `word3 - word1 + word2 = ?` (e.g. king - man + woman = queen) |

### `CBOW(embedding_dim)`

Same interface as `SGNS`. The `train` method additionally accepts `window_size` (fixed context window, default 2).

### `preprocessing`

| Function | Description |
|----------|-------------|
| `load_vocab(data, min_count)` | Load corpus, build vocabulary, compute word frequencies and negative sampling probabilities (unigram^0.75) |
| `get_samples_faster(words, word2id, frequency)` | Generate all (center, target) pairs for SGNS as NumPy arrays. Applies subsampling of frequent words |
| `get_cbow_samples(words, word2id, frequency, window_size)` | Generate all (target, context_words) pairs for CBOW. Fixed window size, skips incomplete windows |

## Usage

**Train:**
```bash
python train.py
```
Trains SGNS (CPU + GPU, dims 50/100/300) and CBOW (GPU, dim 300). Saves embeddings to `embeddings/` and loss/timing data to `training_results.json`.

**Evaluate:**
```bash
python evaluate.py
```
Generates plots to `plots/`, then drops into an interactive word explorer — type comma-separated words to get a side-by-side t-SNE visualization (SGNS vs CBOW).

**Quick usage in code:**
```python
from word2vec import SGNS

model = SGNS(embedding_dim=300)
model.load_embeddings("embeddings/embeddings_sgns_300d.npy")

model.find_similar_words("king", n=5)
model.find_analogy("man", "woman", "king")  # king - man + woman = ?
```

## Setup

```bash
pip install -r requirements.txt
```

CuPy requires an NVIDIA GPU with CUDA. If unavailable, train with `gpu=False` (CPU only).

## Gradient Derivations

Handwritten derivations for all gradients (SGNS positive/negative terms, CBOW chain rule through averaging) are in `resources/`.
