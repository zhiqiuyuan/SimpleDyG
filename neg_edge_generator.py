import argparse
import os
import time
import numpy as np
import torch
from torch_geometric.data import TemporalData
from tqdm import tqdm
from typing import Callable, Optional

from tgb.linkproppred.negative_generator import NegativeEdgeGenerator
from tgb.utils.utils import save_pkl
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

from adapt_dataset import SimpleDyGLinkPropPredDataset, data_max_timestep_map

class NonContinuousNodeIdNegativeEdgeGenerator(NegativeEdgeGenerator):
    def __init__(self, data: TemporalData, dataset_name: str, first_dst_id: int = -1, last_dst_id: int = -1, num_neg_e: int = 100, 
                 strategy: str = "rnd", rnd_seed: int = 123, hist_ratio: float = 0.5, historical_data: TemporalData = None) -> None:
        super().__init__(dataset_name, first_dst_id, last_dst_id, num_neg_e, strategy, rnd_seed, hist_ratio, historical_data)
        self.data = data

    def generate_negative_samples(self, 
                                  data: TemporalData, 
                                  split_mode: str, 
                                  partial_path: str,
                                  ) -> None:
        r"""
        Generate negative samples
        write to filename = (
            partial_path
            + "/"
            + self.dataset_name
            + "_"
            + split_mode
            + "_"
            + "ns"
            + ".pkl"
        )

        Parameters:
            data: an object containing positive edges information
            split_mode: specifies whether to generate negative edges for 'validation' or 'test' splits
            partial_path: in which directory save the generated negatives
        """
        # file name for saving or loading...
        filename = (
            partial_path
            + "/"
            + self.dataset_name
            + "_"
            + split_mode
            + "_"
            + "ns"
            + ".pkl"
        )

        if self.strategy == "rnd":
            self.generate_negative_samples_rnd(data, split_mode, filename)
        elif self.strategy == "hist_rnd":
            self.generate_negative_samples_hist_rnd(
                self.historical_data, data, split_mode, filename
            )
        else:
            raise ValueError("Unsupported negative sample generation strategy!")


    def generate_negative_samples_rnd(self, 
                                      data: TemporalData, 
                                      split_mode: str, 
                                      filename: str,
                                      ) -> None:
        r"""
        Generate negative samples based on the `RND` strategy:
            - for each positive edge, sample a batch of negative edges from all possible edges with the same source node
            - filter actual positive edges
        
        Parameters:
            data: an object containing positive edges information
            split_mode: specifies whether to generate negative edges for 'validation' or 'test' splits
            filename: name of the file containing the generated negative edges
        """
        print(
            f"INFO: Negative Sampling Strategy: {self.strategy}, Data Split: {split_mode}"
        )
        assert split_mode in [
            "val",
            "test",
        ], "Invalid split-mode! It should be `val` or `test`!"

        if os.path.exists(filename):
            print(
                f"INFO: negative samples for '{split_mode}' evaluation are already generated!"
            )
        else:
            print(f"INFO: Generating negative samples for '{split_mode}' evaluation!")
            # retrieve the information from the batch
            pos_src, pos_dst, pos_timestamp = (
                data.src.cpu().numpy(),
                data.dst.cpu().numpy(),
                data.t.cpu().numpy(),
            )

            # all possible destinations
            all_dst = self.data.dst.cpu().numpy()

            evaluation_set = {}
            # generate a list of negative destinations for each positive edge
            pos_edge_tqdm = tqdm(
                zip(pos_src, pos_dst, pos_timestamp), total=len(pos_src)
            )
            for (
                pos_s,
                pos_d,
                pos_t,
            ) in pos_edge_tqdm:
                t_mask = pos_timestamp == pos_t
                src_mask = pos_src == pos_s
                fn_mask = np.logical_and(t_mask, src_mask)
                pos_e_dst_same_src = pos_dst[fn_mask]
                filtered_all_dst = np.setdiff1d(all_dst, pos_e_dst_same_src)

                '''
                when num_neg_e is larger than all possible destinations simple return all possible destinations
                '''
                if (self.num_neg_e > len(filtered_all_dst)):
                    neg_d_arr = filtered_all_dst
                else:
                    neg_d_arr = np.random.choice(
                    filtered_all_dst, self.num_neg_e, replace=False) #never replace negatives

                evaluation_set[(pos_s, pos_d, pos_t)] = neg_d_arr

            # save the generated evaluation set to disk
            save_pkl(evaluation_set, filename)

    def generate_negative_samples_hist_rnd(
        self, 
        historical_data : TemporalData, 
        data: TemporalData, 
        split_mode: str, 
        filename: str,
    ) -> None:
        r"""
        Generate negative samples based on the `HIST-RND` strategy:
            - up to 50% of the negative samples are selected from the set of edges seen during the training with the same source node.
            - the rest of the negative edges are randomly sampled with the fixed source node.

        Parameters:
            historical_data: contains the history of the observed positive edges including 
                            distinct positive edges and edges observed for each positive node
            data: an object containing positive edges information
            split_mode: specifies whether to generate negative edges for 'validation' or 'test' splits
            filename: name of the file to save generated negative edges
        
        Returns:
            None
        """
        print(
            f"INFO: Negative Sampling Strategy: {self.strategy}, Data Split: {split_mode}"
        )
        assert split_mode in [
            "val",
            "test",
        ], "Invalid split-mode! It should be `val` or `test`!"

        if os.path.exists(filename):
            print(
                f"INFO: negative samples for '{split_mode}' evaluation are already generated!"
            )
        else:
            print(f"INFO: Generating negative samples for '{split_mode}' evaluation!")
            # retrieve the information from the batch
            pos_src, pos_dst, pos_timestamp = (
                data.src.cpu().numpy(),
                data.dst.cpu().numpy(),
                data.t.cpu().numpy(),
            )

            pos_ts_edge_dict = {} #{ts: {src: [dsts]}}
            pos_edge_tqdm = tqdm(
                zip(pos_src, pos_dst, pos_timestamp), total=len(pos_src)
            )
            for (
                pos_s,
                pos_d,
                pos_t,
            ) in pos_edge_tqdm:
                if (pos_t not in pos_ts_edge_dict):
                    pos_ts_edge_dict[pos_t] = {pos_s: [pos_d]}
                else:
                    if (pos_s not in pos_ts_edge_dict[pos_t]):
                        pos_ts_edge_dict[pos_t][pos_s] = [pos_d]
                    else:
                        pos_ts_edge_dict[pos_t][pos_s].append(pos_d)

            # all possible destinations
            all_dst = self.data.dst.cpu().numpy()

            # get seen edge history
            (
                historical_edges,
                hist_edge_set_per_node,
            ) = self.generate_historical_edge_set(historical_data)

            # sample historical edges
            max_num_hist_neg_e = int(self.num_neg_e * self.hist_ratio)

            evaluation_set = {}
            # generate a list of negative destinations for each positive edge
            pos_edge_tqdm = tqdm(
                zip(pos_src, pos_dst, pos_timestamp), total=len(pos_src)
            )
            for (
                pos_s,
                pos_d,
                pos_t,
            ) in pos_edge_tqdm:
                pos_e_dst_same_src = np.array(pos_ts_edge_dict[pos_t][pos_s])

                # sample historical edges
                num_hist_neg_e = 0
                neg_hist_dsts = np.array([])
                seen_dst = []
                if pos_s in hist_edge_set_per_node:
                    seen_dst = hist_edge_set_per_node[pos_s]
                    if len(seen_dst) >= 1:
                        filtered_all_seen_dst = np.setdiff1d(seen_dst, pos_e_dst_same_src)
                        #filtered_all_seen_dst = seen_dst #! no collision check
                        num_hist_neg_e = (
                            max_num_hist_neg_e
                            if max_num_hist_neg_e <= len(filtered_all_seen_dst)
                            else len(filtered_all_seen_dst)
                        )
                        neg_hist_dsts = np.random.choice(
                            filtered_all_seen_dst, num_hist_neg_e, replace=False
                        )

                # sample random edges
                if (len(seen_dst) >= 1):
                    invalid_dst = np.concatenate((np.array(pos_e_dst_same_src), seen_dst))
                else:
                    invalid_dst = np.array(pos_e_dst_same_src)
                filtered_all_rnd_dst = np.setdiff1d(all_dst, invalid_dst)

                num_rnd_neg_e = self.num_neg_e - num_hist_neg_e
                '''
                when num_neg_e is larger than all possible destinations simple return all possible destinations
                '''
                if (num_rnd_neg_e > len(filtered_all_rnd_dst)):
                    neg_rnd_dsts = filtered_all_rnd_dst
                else:
                    neg_rnd_dsts = np.random.choice(
                    filtered_all_rnd_dst, num_rnd_neg_e, replace=False
                )
                # concatenate the two sets: historical and random
                neg_dst_arr = np.concatenate((neg_hist_dsts, neg_rnd_dsts))
                evaluation_set[(pos_s, pos_d, pos_t)] = neg_dst_arr

            # save the generated evaluation set to disk
            save_pkl(evaluation_set, filename)

class SimpleDyGPyGLinkPropPredDataset(PyGLinkPropPredDataset):
    def __init__(
        self,
        name: str,
        root: str,
        # transform: Optional[Callable] = None,
        # pre_transform: Optional[Callable] = None,
    ):
        r"""
        PyG wrapper for the LinkPropPredDataset
        can return pytorch tensors for src,dst,t,msg,label
        can return Temporal Data object
        Parameters:
            name: name of the dataset, passed to `LinkPropPredDataset`
            root (string): Root directory where the dataset should be saved, passed to `LinkPropPredDataset`
            transform (callable, optional): A function/transform that takes in an, not used in this case
            pre_transform (callable, optional): A function/transform that takes in, not used in this case
        """
        self.name = name
        self.root = root
        self.dataset = SimpleDyGLinkPropPredDataset(name=name, root=root, preprocess=True)
        self._train_mask = torch.from_numpy(self.dataset.train_mask)
        self._val_mask = torch.from_numpy(self.dataset.val_mask)
        self._test_mask = torch.from_numpy(self.dataset.test_mask)
        # super().__init__(root, transform, pre_transform)
        self._node_feat = self.dataset.node_feat
        self._edge_type = None
        self._static_data = None

        if self._node_feat is None:
            self._node_feat = None
        else:
            self._node_feat = torch.from_numpy(self._node_feat).float()
        
        self.process_data()

        self._ns_sampler = self.dataset.negative_sampler


def main(args):
    r"""
    Generate negative edges for the validation or test phase
    """
    print(f"*** Negative Sample Generation for {args.dataset_name}***")

    num_neg_e_per_pos = args.num_neg_e_per_pos
    neg_sample_strategy = args.neg_sample_strategy
    rnd_seed = args.rnd_seed
    name = args.dataset_name
    assert name in data_max_timestep_map, f'{name} is not in data_max_timestep_map, keys: {data_max_timestep_map.keys()}'
    dataset_folder = os.path.join(args.dataset_root_folder, name, str(data_max_timestep_map[name]))
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder, exist_ok=True)

    print(f'load dataset {name} ...')
    dataset = SimpleDyGPyGLinkPropPredDataset(name=name, root=args.dataset_root_folder)
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    data = dataset.get_TemporalData()

    data_splits = {}
    data_splits['train'] = data[train_mask]
    data_splits['val'] = data[val_mask]
    data_splits['test'] = data[test_mask]

    # After successfully loading the dataset...
    if neg_sample_strategy == "hist_rnd":
        historical_data = data_splits["train"]
    else:
        historical_data = None

    neg_sampler = NonContinuousNodeIdNegativeEdgeGenerator(
        dataset_name=name,
        data=data,
        num_neg_e=num_neg_e_per_pos,
        strategy=neg_sample_strategy,
        rnd_seed=rnd_seed,
        historical_data=historical_data,
    )

    # generate validation negative edge set
    start_time = time.time()
    split_mode = "val"
    print(
        f"INFO: Start generating negative samples: {split_mode} --- {neg_sample_strategy}"
    )
    neg_sampler.generate_negative_samples(
        data=data_splits[split_mode], split_mode=split_mode, partial_path=dataset_folder
    ) 
    print(
        f"INFO: End of negative samples generation. Elapsed Time (s): {time.time() - start_time: .4f}"
    )

    # generate test negative edge set
    start_time = time.time()
    split_mode = "test"
    print(
        f"INFO: Start generating negative samples: {split_mode} --- {neg_sample_strategy}"
    )
    neg_sampler.generate_negative_samples(
        data=data_splits[split_mode], split_mode=split_mode, partial_path=dataset_folder
    )
    print(
        f"INFO: End of negative samples generation. Elapsed Time (s): {time.time() - start_time: .4f}"
    )

class ArgsParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--dataset_name", type=str, default='UCI_13', help="dataset_name"
        )          
        parser.add_argument(
            "--num_neg_e_per_pos", type=int, default=20, help="num_neg_e_per_pos"
        )          
        parser.add_argument(
            "--neg_sample_strategy", type=str, default='hist_rnd', help="neg_sample_strategy"
        )          
        parser.add_argument(
            "--rnd_seed", type=int, default=42, help="rnd_seed"
        )          
        parser.add_argument(
            "--dataset_root_folder", type=str, default='./datasets', help="dataset_root_folder"
        )          

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

'''
python neg_edge_generator.py --dataset_name dialog
python neg_edge_generator.py --dataset_name __ALL__
'''
if __name__ == "__main__":
    args = ArgsParser().parse()
    if args.dataset_name == '__ALL__':
        for name in data_max_timestep_map:
            args.dataset_name = name
            main(args)
    else:
        main(args)
