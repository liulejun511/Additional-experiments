import argparse
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='output prefix, e.g. output/NYT/HTM_K10-50-200_1th_current')
    parser.add_argument('--topk', type=int, default=15)
    return parser.parse_args()


def avg_pairwise_cosine(emb):
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sim = np.matmul(emb, emb.T)
    n = sim.shape[0]
    if n < 2:
        return 0.0
    mask = np.triu(np.ones((n, n), dtype=bool), 1)
    return float(np.mean(sim[mask]))


def avg_topword_overlap(beta, topk=15):
    top_idx = np.argsort(beta, axis=1)[:, -topk:]
    k = top_idx.shape[0]
    if k < 2:
        return 0.0
    overlaps = []
    for i in range(k):
        for j in range(i + 1, k):
            a = set(top_idx[i].tolist())
            b = set(top_idx[j].tolist())
            overlaps.append(len(a & b) / float(topk))
    return float(np.mean(overlaps))


def effective_rank_and_spectral_entropy(mat):
    s = np.linalg.svd(mat, compute_uv=False)
    p = s / (np.sum(s) + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    effective_rank = float(np.exp(entropy))
    return effective_rank, float(entropy)


def main():
    args = parse_args()
    params = np.load(f'{args.path}_params.npz', allow_pickle=True)
    emb_mat = np.load(f'{args.path}_embeddings.npz', allow_pickle=True)

    beta_list = params['beta_list']
    topic_embeddings_list = emb_mat['topic_embeddings_list']

    layer_metrics = []
    for layer_id in range(len(beta_list)):
        beta = beta_list[layer_id]
        topic_emb = topic_embeddings_list[layer_id]

        cosine = avg_pairwise_cosine(topic_emb)
        overlap = avg_topword_overlap(beta, topk=args.topk)
        eff_rank, spec_entropy = effective_rank_and_spectral_entropy(topic_emb)

        layer_metrics.append({
            'layer': int(layer_id),
            'mean_pairwise_cosine': float(cosine),
            'topword_overlap_ratio': float(overlap),
            'effective_rank': float(eff_rank),
            'spectral_entropy': float(spec_entropy)
        })

    overall = {
        'mean_pairwise_cosine': float(np.mean([x['mean_pairwise_cosine'] for x in layer_metrics])),
        'topword_overlap_ratio': float(np.mean([x['topword_overlap_ratio'] for x in layer_metrics])),
        'effective_rank': float(np.mean([x['effective_rank'] for x in layer_metrics])),
        'spectral_entropy': float(np.mean([x['spectral_entropy'] for x in layer_metrics]))
    }
    out = {'layers': layer_metrics, 'overall': overall}

    with open(f'{args.path}_collapse.json', 'w', encoding='utf-8') as file:
        json.dump(out, file, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
