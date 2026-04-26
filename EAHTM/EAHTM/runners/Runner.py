import json
import os
import time
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from utils.model import model_utils
from models.HTM import HTM


def _matrix_entropy(p):
    """Shannon entropy of a coupling matrix (unnormalized ok); scalar."""
    p = p.detach().float().clamp_min(1e-20)
    p = p / p.sum()
    return float(-(p * p.log()).sum().item())


class Runner:
    def __init__(self, args):
        self.args = args
        self.model = HTM(args)
        self.model = self.model.to(args.device)

    def make_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.args.training.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def train(self, dataset_handler, training_stats_path=None):
        optimizer = self.make_optimizer()

        data_size = len(dataset_handler.train_loader.dataset)
        epoch_times = []
        device = self.args.device

        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_train0 = time.perf_counter()
        for epoch in tqdm(range(1, self.args.training.epochs + 1), leave=False):
            t0 = time.perf_counter()
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_data in dataset_handler.train_loader:

                rst_dict = self.model(batch_data)
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            epoch_times.append(time.perf_counter() - t0)

            output_log = f'Epoch: {epoch:03d}'
            for key in loss_rst_dict:
                output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

            print(output_log)

        total_sec = time.perf_counter() - t_train0

        stats = {
            'total_train_sec': total_sec,
            'epochs': int(self.args.training.epochs),
            'mean_epoch_sec': float(np.mean(epoch_times)) if epoch_times else 0.0,
            'std_epoch_sec': float(np.std(epoch_times)) if epoch_times else 0.0,
            'last_epoch_sec': float(epoch_times[-1]) if epoch_times else 0.0,
        }
        if device == 'cuda' and torch.cuda.is_available():
            stats['peak_cuda_allocated_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            stats['peak_cuda_reserved_mb'] = torch.cuda.max_memory_reserved() / (1024 ** 2)

        if getattr(self.args, 'log_ot_stats', False):
            self.model.eval()
            with torch.no_grad():
                batch = next(iter(dataset_handler.train_loader))
                _ = self.model(batch)
                transp_list = getattr(self.model, 'transp_list', None)
                if transp_list is not None and len(transp_list) > 0:
                    stats['ot_transport_entropies'] = [_matrix_entropy(t) for t in transp_list]
                    stats['ot_transport_sparsity'] = [
                        float((t.detach() > 1e-8).float().mean().item()) for t in transp_list
                    ]

        if training_stats_path:
            _d = os.path.dirname(training_stats_path)
            if _d:
                os.makedirs(_d, exist_ok=True)
            with open(training_stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)

        beta_list = self.model.get_beta()
        beta_array = model_utils.np_tensor_list(beta_list)

        return beta_array

    def test(self, input_data):
        data_size = input_data.shape[0]

        hierarchical_theta_list = np.empty(len(self.args.num_topic_list), object)
        for layer_id in range(len(self.args.num_topic_list)):
            hierarchical_theta_list[layer_id] = np.zeros((data_size, self.args.num_topic_list[layer_id]))

        all_idx = torch.split(torch.arange(data_size), self.args.training.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = input_data[idx]
                batch_theta_list = self.model.get_theta(batch_input)

                for layer_id in range(len(self.args.num_topic_list)):
                    hierarchical_theta_list[layer_id][idx] = batch_theta_list[layer_id].cpu().numpy()

        return hierarchical_theta_list
