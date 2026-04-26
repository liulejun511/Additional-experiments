"""
Grid over model.sinkhorn_alpha (Sinkhorn temperature in EA.py: K = exp(-M * alpha)).

Each run invokes run_HTM.py with --sinkhorn_alpha and optional --log_ot_stats.

Example:
  python -m experiments.sinkhorn_sweep --dataset 20NG --alphas 5 10 20 40 80 --test_index 1 --log_training_stats --log_ot_stats
"""
import argparse
import os
import subprocess
import sys

EAHTM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', default='20NG')
    p.add_argument('-k', '--num_topic_str', default='10-50-200')
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--test_index', type=int, default=1)
    p.add_argument('--alphas', type=float, nargs='+', default=[5.0, 10.0, 20.0, 40.0, 80.0])
    p.add_argument('--log_training_stats', action='store_true')
    p.add_argument('--log_ot_stats', action='store_true')
    p.add_argument('--model_config', default='HTM')
    return p.parse_args()


def main():
    args = parse_args()
    root = sys.executable
    script = os.path.join(EAHTM_ROOT, 'run_HTM.py')
    for a in args.alphas:
        cmd = [
            root,
            script,
            '-d', args.dataset,
            '-k', args.num_topic_str,
            '--data_dir', args.data_dir,
            '--test_index', str(args.test_index),
            '-m', args.model_config,
            '--sinkhorn_alpha', str(a),
        ]
        if args.log_training_stats:
            cmd.append('--log_training_stats')
        if args.log_ot_stats:
            cmd.append('--log_ot_stats')
        print('>>>', ' '.join(cmd))
        subprocess.run(cmd, check=True, cwd=EAHTM_ROOT)


if __name__ == '__main__':
    main()
