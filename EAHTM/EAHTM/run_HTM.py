"""
EAHTM 训练入口：读 configs + data，训练 HTM，写出主题词 / theta / phi / 可选训练统计与 OT 诊断。
工作目录应为 EAHTM/EAHTM（与 utils、runners 同级）。详见上层 EAHTM/README.md。
"""
import os
import numpy as np
import argparse
import torch

from utils.data.TextData import TextData
from utils.data import file_utils
from utils.model import model_utils
from runners.Runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='20NG')
    parser.add_argument('-m', '--model_config', default='HTM')
    parser.add_argument('-k', '--num_topic_str', type=str, default='10-50-200')
    parser.add_argument('--num_top_word', type=int, default=15)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--test_index', type=int, default=1)
    parser.add_argument(
        '--sinkhorn_alpha',
        type=float,
        default=None,
        help='Override configs/*.yaml model.sinkhorn_alpha (Sinkhorn temperature).',
    )
    parser.add_argument(
        '--word_embeddings_npz',
        type=str,
        default=None,
        help='Optional path to word_embeddings.npz (same sparse/dense format as data/<ds>/word_embeddings.npz).',
    )
    parser.add_argument(
        '--log_training_stats',
        action='store_true',
        help='Write output/.../training_stats.json (wall time, mean epoch time, peak CUDA memory).',
    )
    parser.add_argument(
        '--log_ot_stats',
        action='store_true',
        help='Append OT transport entropy & sparsity (one train batch, after training) to training_stats.',
    )
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

    args = file_utils.update_args(args, path=f'./configs/{args.model_config}.yaml')

    if getattr(args, 'sinkhorn_alpha', None) is not None:
        args.model.sinkhorn_alpha = float(args.sinkhorn_alpha)

    output_prefix = f'output/{args.dataset}/{args.model_config}_K{args.num_topic_str}_{args.test_index}th'
    file_utils.make_dir(os.path.dirname(output_prefix))

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.num_topic_list = [int(item) for item in args.num_topic_str.split('-')]
    args.num_topic = args.num_topic_list[-1]
    args.num_topic_layers = len(args.num_topic_list)

    dataset_path = f'{args.data_dir}/{args.dataset}'
    dataset_handler = TextData(
        dataset_path,
        args.training.batch_size,
        args.device,
        word_embeddings_npz=getattr(args, 'word_embeddings_npz', None),
    )

    args.vocab_size = dataset_handler.train_data.shape[1]
    args.word_embeddings = dataset_handler.word_embeddings

    runner = Runner(args)

    stats_path = f'{output_prefix}_training_stats.json' if getattr(args, 'log_training_stats', False) else None
    if getattr(args, 'log_ot_stats', False) and not stats_path:
        stats_path = f'{output_prefix}_training_stats.json'

    beta_list = runner.train(dataset_handler, training_stats_path=stats_path)

    topic_str_list = list()
    for layer_id in range(len(beta_list)):
        topic_str_list.extend(export_beta(beta_list[layer_id], dataset_handler.vocab, layer_id, args.num_top_word))

    file_utils.save_text(topic_str_list, f'{output_prefix}_T{args.num_top_word}')

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

    topic_embeddings_list_np = [emb.detach().cpu().numpy() for emb in runner.model.embeddings_list]
    word_embeddings_np = runner.model.bottom_word_embeddings.detach().cpu().numpy()

    np.savez_compressed(
        f'{output_prefix}_embeddings.npz',
        topic_embeddings_list=np.array(topic_embeddings_list_np, dtype=object),
        word_embeddings=word_embeddings_np
    )


if __name__ == '__main__':
    main()
