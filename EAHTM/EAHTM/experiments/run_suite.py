"""
Paper rebuttal experiment launcher (run from EAHTM directory).

  python -m experiments.run_suite --help
  python -m experiments.run_suite nmf --dataset 20NG
  python -m experiments.run_suite collapse --path output/20NG/HTM_K10-50-200_1th
  python -m experiments.run_suite sinkhorn --dataset 20NG --alphas 5 10 20 40
  python -m experiments.run_suite qualitative --path output/20NG/HTM_K10-50-200_1th
  python -m experiments.run_suite ctm_prep --dataset 20NG
  python -m experiments.run_suite topmost --topmost_root D:/code/TopMost-main
  python -m experiments.run_suite fasttext_prep --dataset 20NG --vectors PATH/to/cc.en.300.bin --binary

Embedding ablation (manual): build fastText matrix then train::

  python -m experiments.build_fasttext_embeddings -d 20NG --vectors ... --binary
  python run_HTM.py -d 20NG --word_embeddings_npz ./data/20NG/word_embeddings.fasttext.npz --log_training_stats

EAHTM full train + stats + OT diagnostics::

  python run_HTM.py -d 20NG --log_training_stats --log_ot_stats
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(mod: str, extra: list):
    cmd = [sys.executable, '-m', f'experiments.{mod}', *extra]
    print('>>>', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    p = argparse.ArgumentParser(description='EAHTM paper supplement experiments')
    sub = p.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('nmf', help='TF-IDF + NMF baseline')
    p1.add_argument('-d', '--dataset', required=True)
    p1.add_argument('--data_dir', default='./data')
    p1.add_argument('-k', '--num_topics', type=int, default=200)
    p1.add_argument('--test_index', type=int, default=1)

    p2 = sub.add_parser('collapse', help='Collapse metrics from *_embeddings.npz')
    p2.add_argument('--path', required=True)
    p2.add_argument('--dataset')
    p2.add_argument('--data_dir', default='./data')
    p2.add_argument('--num_top_words', type=int, default=15)

    p3 = sub.add_parser('sinkhorn', help='Sinkhorn alpha grid')
    p3.add_argument('-d', '--dataset', default='20NG')
    p3.add_argument('--data_dir', default='./data')
    p3.add_argument('-k', '--num_topic_str', default='10-50-200')
    p3.add_argument('--test_index', type=int, default=1)
    p3.add_argument('--alphas', type=float, nargs='+', default=[5, 10, 20, 40, 80])
    p3.add_argument('--log_training_stats', action='store_true')
    p3.add_argument('--log_ot_stats', action='store_true')

    p4 = sub.add_parser('qualitative', help='Export hierarchy JSON')
    p4.add_argument('--path', required=True)
    p4.add_argument('--num_top_words', type=int, default=15)

    p5 = sub.add_parser('ctm_prep', help='Prepare files for CTM / CombinedTM')
    p5.add_argument('-d', '--dataset', required=True)
    p5.add_argument('--data_dir', default='./data')

    p6 = sub.add_parser('topmost', help='List TopMost scripts for efficiency runs')
    p6.add_argument('--topmost_root', required=True)

    p7 = sub.add_parser('fasttext_prep', help='Build word_embeddings from fastText file')
    p7.add_argument('-d', '--dataset', required=True)
    p7.add_argument('--data_dir', default='./data')
    p7.add_argument('--vectors', required=True)
    p7.add_argument('--binary', action='store_true')

    args = p.parse_args()

    if args.cmd == 'nmf':
        _run('nmf_baseline', ['-d', args.dataset, '--data_dir', args.data_dir, '-k', str(args.num_topics), '--test_index', str(args.test_index)])
    elif args.cmd == 'collapse':
        ex = ['--path', args.path, '--data_dir', args.data_dir, '--num_top_words', str(args.num_top_words)]
        if args.dataset:
            ex += ['--dataset', args.dataset]
        _run('collapse_diagnostics', ex)
    elif args.cmd == 'sinkhorn':
        ex = [
            '--dataset', args.dataset,
            '--data_dir', args.data_dir,
            '-k', args.num_topic_str,
            '--test_index', str(args.test_index),
            '--alphas', *[str(x) for x in args.alphas],
        ]
        if args.log_training_stats:
            ex.append('--log_training_stats')
        if args.log_ot_stats:
            ex.append('--log_ot_stats')
        _run('sinkhorn_sweep', ex)
    elif args.cmd == 'qualitative':
        _run('export_qualitative', ['--path', args.path, '--num_top_words', str(args.num_top_words)])
    elif args.cmd == 'ctm_prep':
        _run('ctm_baseline', ['-d', args.dataset, '--data_dir', args.data_dir])
    elif args.cmd == 'topmost':
        _run('topmost_efficiency', ['--topmost_root', args.topmost_root])
    elif args.cmd == 'fasttext_prep':
        _run(
            'build_fasttext_embeddings',
            ['-d', args.dataset, '--data_dir', args.data_dir, '--vectors', args.vectors]
            + (['--binary'] if args.binary else []),
        )


if __name__ == '__main__':
    main()
