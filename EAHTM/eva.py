import os
import subprocess
import argparse
import time
from datetime import datetime


def run_script(model, dataset, K, index, read_labels):
    T = 15

    # Print execution information
    print(f"------ {model} {dataset} K={K} {index}th {datetime.now()} ------")

    # Define paths
    prefix = f"./output/{dataset}/{model}_K{K}_{index}th"
    dataset_dir = f"./data"

    # Construct the command
    command = [
        'python',
        'utils/eva/hierarchical_topic_quality.py',
        '--path', prefix,
        '--dataset', dataset,
        '--data_dir', dataset_dir,
        '--read_labels', read_labels
    ]

    try:
        # Print debug information
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True, cwd='D:/study/code/EAHTM/EAHTM')

        print("Script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the script: {e}")
        print(f"Return code: {e.returncode}")
        print(f"Command: {e.cmd}")
        print(f"Output: {e.output}")


if __name__ == "__main__":
    # Parse command line arguments
    model = "HTM"
    dataset = "NYT"
    K = "10-50-200"
    index = 100
    read_labels = 'True'

    # # You can uncomment this block to loop over multiple runs
    # while index < 4:
    #     run_script(model, dataset, K, index, read_labels)
    #     index += 1

    # Execute the script
    run_script(model, dataset, K, index, read_labels)


