import os
import subprocess
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_other_script(model_config, dataset, K, index):
    command = [
        'python',
        os.path.join(_ROOT, 'run_HTM.py'),
        '-m', model_config,
        '-d', dataset,
        '-k', K,
        '--test_index', str(index),
        '--data_dir', os.path.join(_ROOT, 'data'),
    ]
    try:
        subprocess.run(command, check=True, cwd=_ROOT)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the script: {e}")


if __name__ == '__main__':
    model = "HTM"
    dataset = "NYT"
    K = "10-50-200"
    index = 100

    starttime = time.time()
    run_other_script(model, dataset, K, index)
    endtime = time.time()
    print(endtime - starttime)
