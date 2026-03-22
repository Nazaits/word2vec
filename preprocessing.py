from collections import Counter
import time
import random
import numpy as np

def load_vocab(data='text8', min_count=5):
    with open(data, 'r') as f:
        text = f.read()

    words = text.split()

    counts = Counter(words)

    vocab = [word for word, count in counts.items() if count >= min_count]

    word2id = {word: i for i, word in enumerate(vocab)}

    frequency = [counts[word] / len(words) for word in vocab]

    neg_sampling_prob = np.array(frequency) ** 0.75
    neg_sampling_prob /= neg_sampling_prob.sum()
    
    return words, vocab, word2id, frequency, neg_sampling_prob

def get_sample(words, word2id, frequency, neg_sampling_prob, batch_size = 1024, threshold=1e-5, k=5):
    centers = []
    target_samples = []
    negative_samples = []
    for i, word in enumerate(words):
        if word in word2id and random.random() > 1 - (threshold / frequency[word2id[word]]) ** 0.5:
            window_size = random.randint(1, 5)
            targets = words[max(0, i - window_size):i] + words[i + 1:min(i + window_size + 1, len(words))]
            targets = [word2id[target] for target in targets if target in word2id]
            
            for target in targets:
                neg_samples = np.random.choice(len(neg_sampling_prob), size=k, p=neg_sampling_prob)
                
                centers.append(word2id[word])
                target_samples.append(target)
                negative_samples.append(neg_samples)

                if len(centers) == batch_size:
                    yield centers, target_samples, negative_samples
                    centers = []
                    target_samples = []
                    negative_samples = []

    if centers:
        yield centers, target_samples, negative_samples


def get_samples_faster(words, word2id, frequency, threshold=1e-5):
    centers = []
    target_samples = []
    for i, word in enumerate(words):
        if word in word2id and random.random() > 1 - (threshold / frequency[word2id[word]]) ** 0.5:
            window_size = random.randint(1, 5)
            targets = words[max(0, i - window_size):i] + words[i + 1:min(i + window_size + 1, len(words))]
            targets = [word2id[target] for target in targets if target in word2id]

            centers.extend([word2id[word]] * len(targets))
            target_samples.extend(targets)
    
    centers = np.array(centers)
    target_samples = np.array(target_samples) 

    return centers, target_samples


def get_cbow_samples(words, word2id, frequency, threshold=1e-5, window_size=2):
    targets = []
    context_words = []
    for i, word in enumerate(words):
        if word in word2id and random.random() > 1 - (threshold / frequency[word2id[word]]) ** 0.5:

            contexts = words[max(0, i - window_size):i] + words[i + 1:min(i + window_size + 1, len(words))]
            contexts = [word2id[context_word] for context_word in contexts if context_word in word2id]
            
            if len(contexts) != window_size*2:
                continue

            targets.append(word2id[word])
            context_words.append(contexts)
    
    targets = np.array(targets)
    context_words = np.array(context_words) 

    return targets, context_words

if __name__ == "__main__":
    words, vocab, word2id, frequency, neg_sampling_prob = load_vocab()

    st=time.time()
    centers, target_samples = get_samples_faster(words, word2id, frequency)
    et = time.time()
    print(f'Time taken: {et - st} seconds')
    print(len(centers))