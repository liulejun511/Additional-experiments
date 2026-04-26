import os
import json
import argparse
import numpy as np
import scipy.sparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer

from utils.data import file_utils
from utils.eva import classification
from utils.eva import clustering
from utils.eva.hierarchical_topic_quality import compute_topic_coherence, compute_TD


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-k', '--num_topic', type=int, default=200)
    parser.add_argument('--num_top_word', type=int, default=15)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--test_index', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=400)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--read_labels', type=str, default='True')
    return parser.parse_args()


def get_top_words(topic_word, vocab, num_top_word=15):
    top_words = []
    for topic_idx in range(topic_word.shape[0]):
        top_idx = np.argsort(topic_word[topic_idx])[-num_top_word:][::-1]
        words = np.array(vocab)[top_idx]
        top_words.append(f'L-0_K-{topic_idx} ' + ' '.join(words))
    return top_words


def main():
    args = parse_args()

    dataset_path = f'{args.data_dir}/{args.dataset}'
    output_prefix = f'output/{args.dataset}/NMF_K{args.num_topic}_{args.test_index}th_tfidf'
    file_utils.make_dir(os.path.dirname(output_prefix))

    train_bow = scipy.sparse.load_npz(f'{dataset_path}/train_bow.npz').astype('float32')
    test_bow = scipy.sparse.load_npz(f'{dataset_path}/test_bow.npz').astype('float32')
    vocab = file_utils.read_text(f'{dataset_path}/vocab.txt')
    corpus = file_utils.read_text(f'{dataset_path}/train_texts.txt')

    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    train_tfidf = tfidf.fit_transform(train_bow)
    test_tfidf = tfidf.transform(test_bow)

    nmf = NMF(
        n_components=args.num_topic,
        init='nndsvda',
        solver='cd',
        beta_loss='frobenius',
        max_iter=args.max_iter,
        random_state=args.random_state
    )
    train_theta = nmf.fit_transform(train_tfidf)
    test_theta = nmf.transform(test_tfidf)
    topic_word = nmf.components_

    # Normalize document-topic distributions for downstream comparability.
    train_theta = train_theta / (train_theta.sum(axis=1, keepdims=True) + 1e-12)
    test_theta = test_theta / (test_theta.sum(axis=1, keepdims=True) + 1e-12)

    top_words = get_top_words(topic_word, vocab, num_top_word=args.num_top_word)
    file_utils.save_text(top_words, f'{output_prefix}_T{args.num_top_word}')

    processed_top_words = [' '.join(item.split()[1:]) for item in top_words]
    TC = compute_topic_coherence(corpus, vocab, processed_top_words)
    TD = compute_TD(processed_top_words)
    TQ = TC * TD

    results = {
        'model': 'NMF_TFIDF',
        'dataset': args.dataset,
        'num_topic': int(args.num_topic),
        'TC': float(TC),
        'TD': float(TD),
        'TQ': float(TQ)
    }

    np.savez_compressed(
        f'{output_prefix}_params.npz',
        beta_list=np.array([topic_word], dtype=object),
        train_theta_list=np.array([train_theta], dtype=object),
        test_theta_list=np.array([test_theta], dtype=object)
    )

    if args.read_labels == 'True':
        train_labels = np.loadtxt(f'{dataset_path}/train_labels.txt', dtype=int)
        test_labels = np.loadtxt(f'{dataset_path}/test_labels.txt', dtype=int)

        cls = classification.evaluate_classification(train_theta, test_theta, train_labels, test_labels)
        clu = clustering.evaluate_clustering(test_theta, test_labels)
        results['classification'] = cls
        results['clustering'] = clu

    with open(f'{output_prefix}_metrics.json', 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)

    print(json.dumps(results, indent=2))
    print(f'Saved topic words: {output_prefix}_T{args.num_top_word}')
    print(f'Saved metrics: {output_prefix}_metrics.json')


if __name__ == '__main__':
    main()
