"""
Build word_embeddings.npz (csr, float32) for a dataset vocab from fastText/GloVe .vec or .bin.
Use with: python run_HTM.py -d 20NG --word_embeddings_npz ./data/20NG/word_embeddings.fasttext.npz

Example (cc.en.300.vec is large; prefer .bin + gensim):
  python -m experiments.build_fasttext_embeddings -d 20NG --data_dir ./data --vectors cc.en.300.bin --binary
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.data import file_utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', required=True)
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--vectors', required=True, help='Path to fastText .vec or .bin (word2vec format).')
    p.add_argument('--binary', action='store_true', help='Set for .bin Facebook format.')
    p.add_argument(
        '--output',
        default=None,
        help='Output .npz path (default: data/<ds>/word_embeddings.fasttext.npz)',
    )
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from gensim.models import KeyedVectors
    except ImportError:
        print('Requires: pip install gensim')
        sys.exit(1)

    dp = os.path.join(args.data_dir, args.dataset)
    vocab = file_utils.read_text(os.path.join(dp, 'vocab.txt'))
    out = args.output or os.path.join(dp, 'word_embeddings.fasttext.npz')

    print('Loading vectors (may take a while)...')
    kv = KeyedVectors.load_word2vec_format(args.vectors, binary=args.binary)

    dim = kv.vector_size
    W = np.zeros((len(vocab), dim), dtype=np.float32)
    found = 0
    for i, w in enumerate(tqdm(vocab, desc='vocab')):
        if w in kv:
            W[i] = kv[w]
            found += 1
    print(f'===> found embeddings: {found}/{len(vocab)}  dim={dim}')

    scipy.sparse.save_npz(out, scipy.sparse.csr_matrix(W))
    print(f'Wrote {out}')
    print('Train EAHTM with: --word_embeddings_npz', out)
    print('Ensure configs/HTM.yaml embedding-related settings match dim', dim, '(word branch uses this size).')


if __name__ == '__main__':
    main()
