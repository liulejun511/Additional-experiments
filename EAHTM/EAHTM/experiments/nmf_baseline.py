"""
论文非神经基线：TF-IDF（sublinear）+ sklearn NMF，单层 K 个主题。

输出与 EAHTM 评估脚本对齐：output/<ds>/NMF_K<k>_<idx>th_{T,params.npz}；
params 中无 phi_list，hierarchical_topic_quality 只报 C_V / TD / TU 及下游（若给标签）。

运行目录：EAHTM/EAHTM。示例：
  python -m experiments.nmf_baseline -d 20NG --data_dir ./data -k 200
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.data import file_utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', required=True)
    p.add_argument('--data_dir', default='./data')
    p.add_argument('-k', '--num_topics', type=int, default=200, help='NMF n_components (match bottom layer K if comparing).')
    p.add_argument('--num_top_word', type=int, default=15)
    p.add_argument('--test_index', type=int, default=1)
    p.add_argument('--random_state', type=int, default=1)
    p.add_argument('--max_iter', type=int, default=400)
    return p.parse_args()


def main():
    args = parse_args()
    dp = os.path.join(args.data_dir, args.dataset)
    vocab = file_utils.read_text(os.path.join(dp, 'vocab.txt'))
    V = len(vocab)

    train = scipy.sparse.load_npz(os.path.join(dp, 'train_bow.npz')).astype('float32')
    test = scipy.sparse.load_npz(os.path.join(dp, 'test_bow.npz')).astype('float32')

    tfidf = TfidfTransformer(sublinear_tf=True)
    Xtr = tfidf.fit_transform(train)
    Xte = tfidf.transform(test)

    nmf = NMF(
        n_components=args.num_topics,
        random_state=args.random_state,
        max_iter=args.max_iter,
        init='nndsvda',
    )
    W_tr = nmf.fit_transform(Xtr)
    W_te = nmf.transform(Xte)
    H = np.asarray(nmf.components_, dtype=np.float32)

    prefix = f'output/{args.dataset}/NMF_K{args.num_topics}_{args.test_index}th'
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    topic_lines = []
    for t in range(args.num_topics):
        idx = np.argsort(-H[t])[: args.num_top_word]
        words = ' '.join(vocab[i] for i in idx)
        topic_lines.append(f'L-0_K-{t} {words}')
    file_utils.save_text(topic_lines, f'{prefix}_T{args.num_top_word}')

    beta = H / (H.sum(axis=1, keepdims=True) + 1e-12)
    train_theta = np.empty(1, dtype=object)
    test_theta = np.empty(1, dtype=object)
    train_theta[0] = W_tr.astype(np.float32)
    test_theta[0] = W_te.astype(np.float32)

    np.savez_compressed(
        f'{prefix}_params.npz',
        beta_list=np.array([beta], dtype=object),
        train_theta_list=train_theta,
        test_theta_list=test_theta,
    )
    print(f'Wrote {prefix}_T{args.num_top_word} and {prefix}_params.npz')
    print('Evaluate: python utils/eva/hierarchical_topic_quality.py --path', prefix,
          '--dataset', args.dataset, '--data_dir', args.data_dir,
          '--num_top_words', str(args.num_top_word), '--read_labels True')


if __name__ == '__main__':
    main()
