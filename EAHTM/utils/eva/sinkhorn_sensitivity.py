import argparse
import json
import subprocess
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--num_topic_str', default='10-50-200')
    parser.add_argument('--test_index', type=int, default=1)
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--model_config', default='HTM')
    parser.add_argument('--epsilon_list', default='0.02,0.05,0.1,0.2,0.5')
    parser.add_argument('--embedding_type', default='current', choices=['current', 'fasttext'])
    parser.add_argument('--embedding_path', default='')
    parser.add_argument('--read_labels', default='False')
    parser.add_argument('--python_bin', default='python')
    return parser.parse_args()


def compute_transport_stats_from_phi(phi_list):
    entropy_list = []
    sparsity_list = []
    for phi in phi_list:
        next_topic_num = phi.shape[1]
        transp = phi / float(next_topic_num)
        p = transp / (np.sum(transp) + 1e-12)
        entropy = -np.sum(p * np.log(p + 1e-12))
        sparsity = np.mean(p < 1e-6)
        entropy_list.append(float(entropy))
        sparsity_list.append(float(sparsity))
    return float(np.mean(entropy_list)), float(np.mean(sparsity_list))


def main():
    args = parse_args()
    epsilon_values = [float(x.strip()) for x in args.epsilon_list.split(',') if x.strip()]
    all_rows = []

    for eps in epsilon_values:
        run_cmd = [
            args.python_bin, 'run_HTM.py',
            '--dataset', args.dataset,
            '--model_config', args.model_config,
            '--num_topic_str', args.num_topic_str,
            '--test_index', str(args.test_index),
            '--data_dir', args.data_dir,
            '--embedding_type', args.embedding_type,
            '--embedding_path', args.embedding_path,
            '--sinkhorn_epsilon', str(eps)
        ]
        subprocess.run(run_cmd, check=True)

        prefix = f'output/{args.dataset}/{args.model_config}_K{args.num_topic_str}_{args.test_index}th_{args.embedding_type}_eps{eps}'
        eva_cmd = [
            args.python_bin, 'utils/eva/hierarchical_topic_quality.py',
            '--path', prefix,
            '--dataset', args.dataset,
            '--data_dir', args.data_dir,
            '--read_labels', args.read_labels,
            '--save_json', 'True'
        ]
        subprocess.run(eva_cmd, check=True)

        metrics_path = f'{prefix}_metrics.json'
        with open(metrics_path, 'r', encoding='utf-8') as file:
            metrics = json.load(file)

        params = np.load(f'{prefix}_params.npz', allow_pickle=True)
        phi_list = params['phi_list']
        transport_entropy, transport_sparsity = compute_transport_stats_from_phi(phi_list)

        row = {
            'sinkhorn_epsilon': eps,
            'TC': metrics.get('TC'),
            'TD': metrics.get('TD'),
            'TQ': metrics.get('TQ'),
            'PCC': metrics.get('PCC'),
            'PCD': metrics.get('PCD'),
            'transport_entropy': transport_entropy,
            'transport_sparsity': transport_sparsity
        }
        all_rows.append(row)
        print(row)

    out_prefix = f'output/{args.dataset}/{args.model_config}_K{args.num_topic_str}_{args.test_index}th_{args.embedding_type}'
    out_json = f'{out_prefix}_sinkhorn_epsilon_sweep.json'
    with open(out_json, 'w', encoding='utf-8') as file:
        json.dump(all_rows, file, indent=2)
    print(f'Saved: {out_json}')


if __name__ == '__main__':
    main()
