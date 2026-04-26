"""
Topic collapse diagnostics: mean pairwise cosine similarity of topic embeddings,
top-word Jaccard overlap, effective rank / spectral entropy per layer.
Run: python -m experiments.collapse_diagnostics --path output/20NG/HTM_K10-50-200_1th
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.data import file_utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--path', required=True, help='Prefix without _embeddings.npz')
    p.add_argument('--num_top_words', type=int, default=15)
    p.add_argument('--dataset')
    p.add_argument('--data_dir', default='./data')
    return p.parse_args()


def mean_cosine_sim(Z):
    """Z: (K, d) row vectors -> mean pairwise cosine of distinct pairs."""
    n = Z.shape[0]
    if n < 2:
        return float('nan')
    zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    g = zn @ zn.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(g[iu]))


def effective_rank_and_spectral_entropy(Z):
    """From topic embedding matrix (K, d): covariance spectrum on centered Z."""
    X = Z - Z.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    ev = (s ** 2).astype(np.float64)
    ev = ev / (ev.sum() + 1e-20)
    mask = ev > 1e-20
    ev = ev[mask]
    h = float(-np.sum(ev * np.log(ev + 1e-20)))
    er = float(np.exp(h))
    return er, h


def jaccard_top_words(topic_word_sets):
    """topic_word_sets: list of sets of top words; mean Jaccard over pairs."""
    n = len(topic_word_sets)
    if n < 2:
        return float('nan')
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = topic_word_sets[i], topic_word_sets[j]
            u = len(a | b)
            if u == 0:
                continue
            vals.append(len(a & b) / u)
    return float(np.mean(vals)) if vals else float('nan')


def topic_word_sets_from_T_file(path_T, layer_id):
    lines = file_utils.read_text(path_T)
    sets = []
    prefix = f'L-{layer_id}_K-'
    for line in lines:
        if not line.startswith(prefix):
            continue
        rest = line.split(' ', 1)[1] if ' ' in line else ''
        words = set(rest.split()[:50])
        sets.append(words)
    return sets


def main():
    args = parse_args()
    emb = np.load(f'{args.path}_embeddings.npz', allow_pickle=True)
    tel = emb['topic_embeddings_list']
    if tel.dtype == object:
        layers = [np.asarray(tel[i], dtype=np.float32) for i in range(len(tel))]
    else:
        layers = [np.asarray(tel, dtype=np.float32)]

    print('=== collapse diagnostics ===')
    for li, Z in enumerate(layers):
        mc = mean_cosine_sim(Z)
        er, se = effective_rank_and_spectral_entropy(Z)
        line = f'layer {li}: K={Z.shape[0]}, mean_cosine_pair={mc:.5f}, effective_rank={er:.4f}, spectral_entropy={se:.4f}'
        jw = float('nan')
        if args.dataset:
            T = args.num_top_words
            pT = f'{args.path}_T{T}'
            if os.path.isfile(pT):
                tws = topic_word_sets_from_T_file(pT, li)
                if len(tws) >= 2:
                    jw = jaccard_top_words(tws)
                    line += f', mean_topword_jaccard={jw:.5f}'
        print(line)


if __name__ == '__main__':
    main()
