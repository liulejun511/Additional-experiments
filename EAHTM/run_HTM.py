import os
import numpy as np
import argparse
import torch
import json
import subprocess

from utils.data.TextData import TextData
from utils.data import file_utils
from utils.model import model_utils
from runners.Runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset')
    parser.add_argument('-m', '--model_config', default='HTM')
    parser.add_argument('-k', '--num_topic_str', type=str, default='10-50-200')
    parser.add_argument('--num_top_word', type=int, default=15)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--test_index', type=int, default=1)
    parser.add_argument('--embedding_type', type=str, default='current', choices=['current', 'fasttext'])
    parser.add_argument('--embedding_path', type=str, default='')
    parser.add_argument('--sinkhorn_epsilon', type=float, default=None)
    parser.add_argument('--sinkhorn_alpha', type=float, default=None)
    parser.add_argument('--run_all_eval', type=str, default='True')
    parser.add_argument('--read_labels', type=str, default='True')
    args = parser.parse_args()
    return args


def export_beta(beta, vocab, layer_id, num_top_word=15):
    topic_str_list = model_utils.print_topic_words(beta, vocab, num_top_word=num_top_word)
    for k, topic_str in enumerate(topic_str_list):
        topic_str_list[k] = f'L-{layer_id}_K-{k} {topic_str}'
        print(topic_str_list[k])
    return topic_str_list


def main():
    args = parse_args()
    print(f"Current working directory: {os.getcwd()}")

    # loading model configuration
    args = file_utils.update_args(args, path=f'./configs/{args.model_config}.yaml')

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.embedding_type = args.embedding_type
    args.embedding_path = args.embedding_path
    if args.sinkhorn_epsilon is not None:
        args.model.sinkhorn_epsilon = args.sinkhorn_epsilon
    if args.sinkhorn_alpha is not None:
        args.model.sinkhorn_alpha = args.sinkhorn_alpha

    # split the string to get a list of topic numbers
    args.num_topic_list = [int(item) for item in args.num_topic_str.split('-')]
    args.num_topic = args.num_topic_list[-1]
    args.num_topic_layers = len(args.num_topic_list)
    sinkhorn_tag = 'default'
    if getattr(args.model, 'sinkhorn_epsilon', None) is not None:
        sinkhorn_tag = f'eps{args.model.sinkhorn_epsilon}'
    elif getattr(args.model, 'sinkhorn_alpha', None) is not None:
        sinkhorn_tag = f'alpha{args.model.sinkhorn_alpha}'

    output_prefix = f'output/{args.dataset}/{args.model_config}_K{args.num_topic_str}_{args.test_index}th_{args.embedding_type}_{sinkhorn_tag}'
    file_utils.make_dir(os.path.dirname(output_prefix))

    dataset_path = f'{args.data_dir}/{args.dataset}'
    dataset_handler = TextData(
        dataset_path,
        args.training.batch_size,
        args.device,
        embedding_type=args.embedding_type,
        embedding_path=args.embedding_path
    )

    args.vocab_size = dataset_handler.train_data.shape[1]
    args.word_embeddings = dataset_handler.word_embeddings

    runner = Runner(args)

    beta_list, train_metrics = runner.train(dataset_handler)

    # print and save topic words.
    topic_str_list = list()
    for layer_id, num_topic in enumerate(range(len(beta_list))):
        topic_str_list.extend(export_beta(beta_list[layer_id], dataset_handler.vocab, layer_id, args.num_top_word))

    file_utils.save_text(topic_str_list, f'{output_prefix}_T{args.num_top_word}')

    # save inferred topic distributions of training set and testing set.
    train_theta_list = runner.test(dataset_handler.train_data)
    test_theta_list = runner.test(dataset_handler.test_data)

    params_dict = {
        'beta_list': beta_list,
        'train_theta_list': train_theta_list,
        'test_theta_list': test_theta_list
    }

    phi_list = runner.model.get_phi_list()
    if isinstance(phi_list[0], torch.Tensor):
        phi_list = model_utils.np_tensor_list(phi_list)
    params_dict['phi_list'] = phi_list
    np.savez_compressed(f'{output_prefix}_params.npz', **params_dict)

    # 转为 numpy
    topic_embeddings_list_np = [emb.detach().cpu().numpy() for emb in runner.model.embeddings_list]
    word_embeddings_np = runner.model.bottom_word_embeddings.detach().cpu().numpy()

    # save
    np.savez_compressed(
        f'{output_prefix}_embeddings.npz',
        topic_embeddings_list=np.array(topic_embeddings_list_np, dtype=object),
        word_embeddings=word_embeddings_np
    )

    with open(f'{output_prefix}_efficiency.json', 'w', encoding='utf-8') as file:
        json.dump(train_metrics, file, indent=2)

    if args.run_all_eval == 'True':
        eval_cmd = [
            'python', 'utils/eva/hierarchical_topic_quality.py',
            '--path', output_prefix,
            '--dataset', args.dataset,
            '--data_dir', args.data_dir,
            '--read_labels', args.read_labels,
            '--save_json', 'True'
        ]
        print(f'Running evaluation: {" ".join(eval_cmd)}')
        subprocess.run(eval_cmd, check=True)

        collapse_cmd = [
            'python', 'utils/eva/collapse_diagnostics.py',
            '--path', output_prefix,
            '--topk', str(args.num_top_word)
        ]
        print(f'Running collapse diagnostics: {" ".join(collapse_cmd)}')
        subprocess.run(collapse_cmd, check=True)

if __name__ == '__main__':
    main()
