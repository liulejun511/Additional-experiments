import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
import time
from utils.model import model_utils
from models.HTM import HTM


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

    def train(self, dataset_handler):
        optimizer = self.make_optimizer()

        data_size = len(dataset_handler.train_loader.dataset)
        epoch_time_list = []
        if torch.cuda.is_available() and self.args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        train_start_time = time.perf_counter()

        for epoch in tqdm(range(1, self.args.training.epochs + 1), leave=False):
            self.model.train()
            loss_rst_dict = defaultdict(float)
            epoch_start_time = time.perf_counter()

            for batch_data in dataset_handler.train_loader:

                rst_dict = self.model(batch_data)
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            output_log = f'Epoch: {epoch:03d}'
            for key in loss_rst_dict:
                output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

            print(output_log)
            epoch_time_list.append(time.perf_counter() - epoch_start_time)

        total_train_time = time.perf_counter() - train_start_time

        beta_list = self.model.get_beta()
        beta_array = model_utils.np_tensor_list(beta_list)

        if torch.cuda.is_available() and self.args.device == 'cuda':
            peak_gpu_mem_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        else:
            peak_gpu_mem_mb = None

        metrics = {
            'train_time_total_sec': float(total_train_time),
            'train_time_per_epoch_sec_mean': float(np.mean(epoch_time_list)) if len(epoch_time_list) > 0 else 0.,
            'train_time_per_epoch_sec_std': float(np.std(epoch_time_list)) if len(epoch_time_list) > 0 else 0.,
            'peak_gpu_mem_mb': peak_gpu_mem_mb,
            'num_params': int(sum(p.numel() for p in self.model.parameters()))
        }

        return beta_array, metrics

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
