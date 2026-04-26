import argparse
import subprocess
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--num_topic_str', default='10-50-200')
    parser.add_argument('--test_index', type=int, default=1)
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--model_config', default='HTM')
    parser.add_argument('--fasttext_path', default='')
    parser.add_argument('--read_labels', default='True')
    parser.add_argument('--python_bin', default='python')
    return parser.parse_args()


def run_one(args, embedding_type, embedding_path=''):
    run_cmd = [
        args.python_bin, 'run_HTM.py',
        '--dataset', args.dataset,
        '--model_config', args.model_config,
        '--num_topic_str', args.num_topic_str,
        '--test_index', str(args.test_index),
        '--data_dir', args.data_dir,
        '--embedding_type', embedding_type,
        '--embedding_path', embedding_path
    ]
    subprocess.run(run_cmd, check=True)

    prefix = f'output/{args.dataset}/{args.model_config}_K{args.num_topic_str}_{args.test_index}th_{embedding_type}_alpha20'
    eva_cmd = [
        args.python_bin, 'utils/eva/hierarchical_topic_quality.py',
        '--path', prefix,
        '--dataset', args.dataset,
        '--data_dir', args.data_dir,
        '--read_labels', args.read_labels,
        '--save_json', 'True'
    ]
    subprocess.run(eva_cmd, check=True)
    with open(f'{prefix}_metrics.json', 'r', encoding='utf-8') as file:
        return prefix, json.load(file)


def main():
    args = parse_args()
    rows = []

    prefix, metrics = run_one(args, embedding_type='current', embedding_path='')
    rows.append({'embedding_type': 'current', 'prefix': prefix, **metrics})

    prefix, metrics = run_one(args, embedding_type='fasttext', embedding_path=args.fasttext_path)
    rows.append({'embedding_type': 'fasttext', 'prefix': prefix, **metrics})

    out_path = f'output/{args.dataset}/{args.model_config}_K{args.num_topic_str}_{args.test_index}th_embedding_sensitivity.json'
    with open(out_path, 'w', encoding='utf-8') as file:
        json.dump(rows, file, indent=2)
    print(json.dumps(rows, indent=2))
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
