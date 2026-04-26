"""
Export hierarchical topic tree as JSON (same as utils/eva/show_topic_hierarchy.py).
Use for qualitative appendix (20NG, NYT, multiple seeds / K configs).

Run from EAHTM root:
  python -m experiments.export_qualitative --path output/20NG/HTM_K10-50-200_1th --num_top_words 15
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.data import file_utils
from utils.model import model_utils
from utils.eva.show_topic_hierarchy import build_hierarchy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--path', required=True, help='Output prefix without _T15 / _params')
    p.add_argument('--num_top_words', type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    tpath = f'{args.path}_T{args.num_top_words}'
    ppath = f'{args.path}_params.npz'
    if not os.path.isfile(tpath) or not os.path.isfile(ppath):
        raise FileNotFoundError(f'Missing {tpath} or {ppath}')

    data_mat = np.load(ppath, allow_pickle=True)
    phi_list = data_mat['phi_list']
    hierarchical_topic_dict = model_utils.convert_topicStr_to_dict(file_utils.read_text(tpath))
    tree = build_hierarchy(hierarchical_topic_dict, phi_list)
    out = f'{args.path}_hierarchy-T{args.num_top_words}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    print('Wrote', os.path.abspath(out))


if __name__ == '__main__':
    main()
