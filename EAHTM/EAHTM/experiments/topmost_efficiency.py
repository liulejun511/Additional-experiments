"""
TopMost (ProdLDA / SawETM) wall-clock & VRAM — run on a clone of TopMost in --topmost_root.

EAHTM side: use `python run_HTM.py ... --log_training_stats` and read `*_training_stats.json`.

This script only checks for common entry files and prints a concrete command template.

Paper / demo: https://aclanthology.org/2024.acl-demos.4/
Repo: https://github.com/bobxwu/topmost

Example:
  python -m experiments.topmost_efficiency --topmost_root D:/code/TopMost-main
"""
import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--topmost_root', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.topmost_root)
    if not os.path.isdir(root):
        print('Not a directory:', root)
        sys.exit(1)

    candidates = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.py') and ('run' in fn.lower() or 'main' in fn.lower() or 'train' in fn.lower()):
                candidates.append(os.path.join(dirpath, fn))
        if len(candidates) > 80:
            break

    print('TopMost root:', root)
    print('Sample Python files (pick your dataset script and wrap with time / torch.cuda.max_memory_allocated):')
    for c in candidates[:25]:
        print(' ', c)
    print()
    print('Match EAHTM: same GPU, batch_size, epoch count (or document ratio in appendix).')
    print('EAHTM timing JSON: run_HTM.py --log_training_stats [--log_ot_stats]')


if __name__ == '__main__':
    main()
