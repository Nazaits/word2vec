import numpy as np
import cupy as cp
from preprocessing import load_vocab, get_samples_faster, get_cbow_samples
import time


class SGNS:
    def __init__(self, embedding_dim):
        self.words, self.vocab, self.word2id, self.frequency, self.neg_sampling_prob = load_vocab()
        
        #embedding and context matrices
        self.E = np.random.randn(len(self.vocab), embedding_dim)*0.01
        self.C = np.random.randn(len(self.vocab), embedding_dim)*0.01

    def sigmoid(self, x, xp): # xp is either np or cp, to avoid importing cupy in the whole file and just use it in this function
        return xp.clip(1 / (1 + xp.exp(-x)), 1e-15, 1 - 1e-15)
        
    def train(self, epochs=1, learning_rate=0.01, k=5, batch_size=1024, gpu=False):
        # Use cupy library for gpu acceleration 
        xp = cp if gpu else np
        self.E = xp.asarray(self.E)
        self.C = xp.asarray(self.C)

        centers, target_samples = get_samples_faster(self.words, self.word2id, self.frequency)
        Ls = []
        for epoch in range(epochs):
            L = 0
            sample = 0
            # shuffle because it's supposedly better
            indices = np.random.permutation(len(centers))
            centers = centers[indices]
            target_samples = target_samples[indices]

            # generate negative samples
            negative_samples = np.random.choice(len(self.word2id), size=(len(centers), k), p=self.neg_sampling_prob)

            for i in range(0, len(centers), batch_size):
                word_ids, target_ids, negative_ids = centers[i:i+batch_size], target_samples[i:i+batch_size], negative_samples[i:i+batch_size]
                # print(sample)
                sample += 1
                v_center = self.E[word_ids]
                v_target = self.C[target_ids]

                v_negatives = self.C[negative_ids]

                # precompute to not call twice
                sigmoid_center_target = self.sigmoid((v_center*v_target).sum(axis=1), xp)
                sigmoid_negatives_center = self.sigmoid(v_negatives@v_center.reshape(*v_center.shape, 1), xp).squeeze(axis=2) # (batch_size, k, dim) @ (batch_size, dim, 1) -> (batch_size, k, 1). @ supports batched matmul 
                
                # Loss, not important for training, just for monitoring
                L += xp.sum(-xp.log(sigmoid_center_target) - xp.sum(xp.log(1 - sigmoid_negatives_center)))

                # look at the paper math for the formulas, but tldr:
                # dv_center = (s(v_center@v_target) - 1)*v_target + sum(s(v_negatives@v_center)*v_negatives)
                # dv_target = (s(v_center@v_target) - 1)*v_center
                # dv_negatives = s(v_negatives@v_center)*v_center
                # the reshaping and multiplication is for parallelization. You can see in the scribble notes how I somehow derived it (although I myself am not sure how I got it anymore)
                dv_center = (sigmoid_center_target - 1).reshape(-1, 1)*v_target + (sigmoid_negatives_center.reshape(sigmoid_negatives_center.shape[0], 1, -1)@v_negatives).squeeze(axis=1) # paralelization of negative samples, avoid summing over
                dv_target = (sigmoid_center_target - 1).reshape(-1, 1)*v_center
                dv_negatives = sigmoid_negatives_center.reshape(sigmoid_negatives_center.shape[0], -1, 1)@v_center.reshape(v_center.shape[0], 1, -1) # (batch_size, k, 1) @ (batch_size, dim, 1) -> (batch_size, k, dim)

                xp.add.at(self.E, word_ids, -learning_rate * dv_center)
                xp.add.at(self.C, target_ids, -learning_rate * dv_target)
                xp.add.at(self.C, negative_ids, -learning_rate * dv_negatives)

            print(f'Epoch: {epoch}, Loss: {L}')
            Ls.append(L)
        
        self.E = self.E.get() if gpu else self.E
        self.C = self.C.get() if gpu else self.C

        return Ls
    
    def save_embeddings(self, file_name='embeddings.npy'):
        np.save(file_name, self.E)
    def load_embeddings(self, file_name='embeddings.npy'):
        self.E = np.load(file_name)

    def find_similar(self, v_word, n=5):
        cosine_similarities = self.E@v_word/(np.linalg.norm(v_word)*np.linalg.norm(self.E, axis=1))

        most_similar = np.argsort(cosine_similarities)[::-1][1:n+1].tolist()

        return np.array(self.vocab)[most_similar]
    
    def find_similar_words(self, word, n=5):
        if word not in self.word2id:
            return []
        
        v_word = self.E[self.word2id[word]]
        return self.find_similar(v_word, n=n)


    def find_analogy(self, word1, word2, word3):
        if word1 not in self.word2id or word2 not in self.word2id or word3 not in self.word2id:
            return []
        
        v_word1 = self.E[self.word2id[word1]]
        v_word2 = self.E[self.word2id[word2]]
        v_word3 = self.E[self.word2id[word3]]

        v_analogy = v_word3 - v_word1 + v_word2

        return self.find_similar(v_analogy, n=5)



# Copy and pasted from SGNS just adapted the train function
class CBOW:
    def __init__(self, embedding_dim):
        self.words, self.vocab, self.word2id, self.frequency, self.neg_sampling_prob = load_vocab()
        
        #embedding and context matrices
        self.E = np.random.randn(len(self.vocab), embedding_dim)*0.01
        self.C = np.random.randn(len(self.vocab), embedding_dim)*0.01

    def sigmoid(self, x, xp): # xp is either np or cp, to avoid importing cupy in the whole file and just use it in this function
        return xp.clip(1 / (1 + xp.exp(-x)), 1e-15, 1 - 1e-15)
    
    
    def train(self, epochs=1, learning_rate=0.01, window_size=2, k=5, batch_size=1024, gpu=False):
        # Use cupy library for gpu acceleration 
        xp = cp if gpu else np
        self.E = xp.asarray(self.E)
        self.C = xp.asarray(self.C)

        targets, context_words = get_cbow_samples(self.words, self.word2id, self.frequency, window_size=window_size)
        Ls = []
        for epoch in range(epochs):
            L = 0
            sample = 0
            # shuffle because it's supposedly better
            indices = np.random.permutation(len(targets))
            targets = targets[indices]
            context_words = context_words[indices]

            # generate negative samples
            negative_samples = np.random.choice(len(self.word2id), size=(len(targets), k), p=self.neg_sampling_prob)

            for i in range(0, len(targets), batch_size):
                word_ids, context_ids, negative_ids = targets[i:i+batch_size], context_words[i:i+batch_size], negative_samples[i:i+batch_size]
                # print(sample)
                sample += 1
                # We take the averages of the context words
                v_center = xp.average(self.E[context_ids], axis=1)
                v_target = self.C[word_ids]

                v_negatives = self.C[negative_ids]

                # precompute to not call twice
                sigmoid_center_target = self.sigmoid((v_center*v_target).sum(axis=1), xp)
                sigmoid_negatives_center = self.sigmoid(v_negatives@v_center.reshape(*v_center.shape, 1), xp).squeeze(axis=2) # (batch_size, k, dim) @ (batch_size, dim, 1) -> (batch_size, k, 1). @ supports batched matmul 
                
                # Loss, not important for training, just for monitoring
                L += xp.sum(-xp.log(sigmoid_center_target) - xp.sum(xp.log(1 - sigmoid_negatives_center)))

                # look at the paper math for the formulas, but tldr:
                # dv_center = (s(v_center@v_target) - 1)*v_target + sum(s(v_negatives@v_center)*v_negatives)
                # dv_target = (s(v_center@v_target) - 1)*v_center
                # dv_negatives = s(v_negatives@v_center)*v_center
                # the reshaping and multiplication is for parallelization. You can see in the scribble notes how I somehow derived it (although I myself am not sure how I got it anymore)
                dv_center = (sigmoid_center_target - 1).reshape(-1, 1)*v_target + (sigmoid_negatives_center.reshape(sigmoid_negatives_center.shape[0], 1, -1)@v_negatives).squeeze(axis=1) # paralelization of negative samples, avoid summing over
                dv_target = (sigmoid_center_target - 1).reshape(-1, 1)*v_center
                dv_negatives = sigmoid_negatives_center.reshape(sigmoid_negatives_center.shape[0], -1, 1)@v_center.reshape(v_center.shape[0], 1, -1) # (batch_size, k, 1) @ (batch_size, dim, 1) -> (batch_size, k, dim)

                # Adjust the broadcasting and updating of E matrix as to acount for the averaging at the start
                xp.add.at(self.E, context_ids, -learning_rate * dv_center.reshape(dv_center.shape[0], 1, -1) * 1/(window_size*2))
                xp.add.at(self.C, word_ids, -learning_rate * dv_target)
                xp.add.at(self.C, negative_ids, -learning_rate * dv_negatives)

            print(f'Epoch: {epoch}, Loss: {L}')
            Ls.append(L)
        
        self.E = self.E.get() if gpu else self.E
        self.C = self.C.get() if gpu else self.C

        return Ls
    
    def save_embeddings(self, file_name='embeddings.npy'):
        np.save(file_name, self.E)
    def load_embeddings(self, file_name='embeddings.npy'):
        self.E = np.load(file_name)

    def find_similar(self, v_word, n=5):
        cosine_similarities = self.E@v_word/(np.linalg.norm(v_word)*np.linalg.norm(self.E, axis=1))

        most_similar = np.argsort(cosine_similarities)[::-1][1:n+1].tolist()

        return np.array(self.vocab)[most_similar]
    
    def find_similar_words(self, word, n=5):
        if word not in self.word2id:
            return []
        
        v_word = self.E[self.word2id[word]]
        return self.find_similar(v_word, n=n)


    def find_analogy(self, word1, word2, word3):
        if word1 not in self.word2id or word2 not in self.word2id or word3 not in self.word2id:
            return []
        
        v_word1 = self.E[self.word2id[word1]]
        v_word2 = self.E[self.word2id[word2]]
        v_word3 = self.E[self.word2id[word3]]

        v_analogy = v_word3 - v_word1 + v_word2

        return self.find_similar(v_analogy, n=5)
