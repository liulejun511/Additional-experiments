import argparse
import json
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True, help='e.g. output/NYT')
    parser.add_argument('--pattern', default='*_efficiency.json')
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    paths = sorted(output_dir.glob(args.pattern))
    if not paths:
        print('No efficiency json files found.')
        return

    rows = []
    for p in paths:
        with open(p, 'r', encoding='utf-8') as file:
            row = json.load(file)
        row['run_name'] = p.stem.replace('_efficiency', '')
        rows.append(row)

    out_json = output_dir / 'efficiency_summary.json'
    with open(out_json, 'w', encoding='utf-8') as file:
        json.dump(rows, file, indent=2)

    out_csv = output_dir / 'efficiency_summary.csv'
    headers = [
        'run_name',
        'train_time_total_sec',
        'train_time_per_epoch_sec_mean',
        'train_time_per_epoch_sec_std',
        'peak_gpu_mem_mb',
        'num_params'
    ]
    lines = [','.join(headers)]
    for r in rows:
        vals = [str(r.get(h, '')) for h in headers]
        lines.append(','.join(vals))
    out_csv.write_text('\n'.join(lines), encoding='utf-8')

    total_times = [r.get('train_time_total_sec', np.nan) for r in rows]
    print(f'Collected {len(rows)} runs')
    print(f'Mean total train time: {np.nanmean(total_times):.4f} sec')
    print(f'Wrote: {out_json}')
    print(f'Wrote: {out_csv}')


if __name__ == '__main__':
    main()
