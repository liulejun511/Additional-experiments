import torch
from torch.utils.data import DataLoader
import scipy.sparse
import scipy.io
import gensim
import numpy as np
from tqdm import tqdm
from utils.data import file_utils
import os

class TextData:
    def __init__(self, dataset_path, batch_size, device, word_embeddings_npz=None):
        """
        word_embeddings_npz: optional path to a .npz (dense csr or array) of shape (|V|, d).
        When set, overrides the default {dataset_path}/word_embeddings.npz lookup.
        """
        self._word_embeddings_npz = word_embeddings_npz
        self.train_data, self.test_data, self.vocab, self.word_embeddings = self.load_data(dataset_path)
        self.vocab_size = len(self.vocab)

        self.train_data = torch.from_numpy(self.train_data).to(device)
        self.test_data = torch.from_numpy(self.test_data).to(device)

        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)

    def load_data(self, path):
        train_data = scipy.sparse.load_npz(f'{path}/train_bow.npz').toarray().astype('float32')
        test_data = scipy.sparse.load_npz(f'{path}/test_bow.npz').toarray().astype('float32')
        vocab = file_utils.read_text(f'{path}/vocab.txt')

        if self._word_embeddings_npz:
            embeddings_path = self._word_embeddings_npz
            if not os.path.isabs(embeddings_path):
                cand = os.path.join(path, embeddings_path)
                embeddings_path = cand if os.path.exists(cand) else os.path.abspath(embeddings_path)
        else:
            embeddings_path = f'{path}/word_embeddings.npz'
        if os.path.exists(embeddings_path):
            word_embeddings = scipy.sparse.load_npz(embeddings_path).toarray().astype('float32')
        else:
            word_embeddings = self.make_word_embeddings(os.path.dirname(path), vocab)
            scipy.sparse.save_npz(embeddings_path, word_embeddings)
        return train_data, test_data, vocab, word_embeddings

    def make_word_embeddings(self, dir_path, vocab):
        vector_path = f'{dir_path}/wordVector/glove.6B.300d.txt'
        glove_vectors = gensim.models.KeyedVectors.load_word2vec_format(vector_path, no_header=True)
        word_embeddings = np.zeros((len(vocab), glove_vectors.vectors.shape[1]))

        num_found = 0
        for i, word in enumerate(tqdm(vocab, desc="===>making word embeddings")):
            try:
                key_word_list = glove_vectors.index_to_key
            except:
                key_word_list = glove_vectors.index2word

            if word in key_word_list:
                word_embeddings[i] = glove_vectors[word]
                num_found += 1

        print(f'===> number of found embeddings: {num_found}/{len(vocab)}')

        return scipy.sparse.csr_matrix(word_embeddings)
