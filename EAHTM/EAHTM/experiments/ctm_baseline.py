"""
Contextualized Topic Models (CombinedTM) — data prep + reproducible file layout.

The `contextualized-topic-models` API changes between releases; this script only
materializes inputs next to EAHTM data (BoW CSR components + texts + vocab).
Train CombinedTM in a small notebook or script using the official repo, then
export top words as flat lines::

    L-0_K-0 w1 w2 ...
    L-0_K-1 ...

and a *_params.npz with train_theta_list (object array of one layer) for
downstream. Evaluate with::

    python utils/eva/hierarchical_topic_quality.py --path output/<ds>/CTM_... \\
        --dataset <ds> --data_dir ./data --skip_hierarchy --read_labels True

Paper: https://aclanthology.org/2021.acl-short.96/
Code: https://github.com/MilaNLProc/contextualized-topic-models

Run: python -m experiments.ctm_baseline -d 20NG --data_dir ./data
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.data import file_utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', required=True)
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--test_index', type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    dp = os.path.join(args.data_dir, args.dataset)
    vocab = file_utils.read_text(os.path.join(dp, 'vocab.txt'))
    texts = file_utils.read_text(os.path.join(dp, 'train_texts.txt'))
    train_bow = scipy.sparse.load_npz(os.path.join(dp, 'train_bow.npz')).tocsr()

    side = f'output/{args.dataset}/CTM_prep_{args.test_index}th'
    os.makedirs(side, exist_ok=True)
    np.save(os.path.join(side, 'train_bow_indices.npy'), train_bow.indices)
    np.save(os.path.join(side, 'train_bow_indptr.npy'), train_bow.indptr)
    np.save(os.path.join(side, 'train_bow_data.npy'), train_bow.data.astype(np.float32))
    np.save(os.path.join(side, 'train_bow_shape.npy'), np.array(train_bow.shape))
    file_utils.save_text(texts, os.path.join(side, 'train_texts_copy.txt'))
    file_utils.save_text(vocab, os.path.join(side, 'vocab_copy.txt'))
    meta = {
        'dataset': args.dataset,
        'num_docs': len(texts),
        'vocab_size': len(vocab),
        'bow_nnz': int(train_bow.nnz),
        'readme': 'Reconstruct train_bow with scipy.sparse.csr_matrix((data, indices, indptr), shape=...)',
    }
    with open(os.path.join(side, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('Prepared CTM sidecar:', os.path.abspath(side))
    print('Install: pip install contextualized-topic-models sentence-transformers')
    print('Train CombinedTM on train_texts + BoW; align C_V/TD reference corpus with train_texts.txt.')


if __name__ == '__main__':
    main()
