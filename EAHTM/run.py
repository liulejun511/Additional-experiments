import subprocess
import time

def run_other_script(model, dataset, K, index):
    # Construct the command line
    command = [
        'python',  # Use python to execute the script
        'D:/study/code/EAHTM/EAHTM/run_HTM.py',  # Path to the target script
        '--model', model,
        '--dataset', dataset,
        '--num_topic_str', K,
        '--test_index', str(index)
    ]

    # Execute the script with the given arguments
    try:
        subprocess.run(command, check=True, cwd='D:/study/code/EAHTM/EAHTM')  # Specify working directory
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the script: {e}")

if __name__ == '__main__':
    # Assume parameters are obtained from command line or elsewhere
    model = "HTM"
    dataset = "NYT"
    K = "10-50-200"
    index = 100

    # Example loop execution (commented out)
    # while index < 4:
    #     run_other_script(model, dataset, K, index)
    #     index += 1

    starttime = time.time()
    run_other_script(model, dataset, K, index)
    endtime = time.time()
    total_time = endtime - starttime
    print(total_time)
