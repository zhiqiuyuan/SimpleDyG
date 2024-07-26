import numpy as np
import pandas as pd
from typing import Optional
import os.path as osp
import sys

from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.linkproppred.evaluate import Evaluator

from utils.DataLoader import Data
from adapt_problem import SimpleDyGStandardize

data_num_nodes_map = {
    'UCI_13': 1781,
    'ML_10M_13': 15841,
    'hepth': 4737,
    # 'dialog': 7415, # value reported in paper
    'dialog': 7472,
}

data_num_edges_map = {
    'UCI_13': 16743,
    'ML_10M_13': 48561,
    'hepth': 14831,
    # 'dialog': 91986, # value reported in paper
    'dialog': 98249,
}

data_max_timestep_map = {
    'UCI_13': 12,
    'ML_10M_13': 12,
    'hepth': 11,
    'dialog': 15,
}

class SimpleDyGLinkPropPredDataset(LinkPropPredDataset):
    def __init__(
        self,
        name: str,
        root: Optional[str] = "datasets",
        meta_dict: Optional[dict] = None,
        preprocess: Optional[bool] = True,
    ):
        r"""Dataset class for link prediction dataset. Stores meta information about each dataset such as evaluation metrics etc.
        also automatically pre-processes the dataset.
        Args:
            name: name of the dataset
            root: root directory to store the dataset folder
            meta_dict: dictionary containing meta information about the dataset, should contain key 'dir_name' which is the name of the dataset folder
            preprocess: whether to pre-process the dataset
        # ! for datasets used by simpledyg paper, number of timesteps are determined by the original csv file (the 'timestamp' column)
        """
        self.name = name  ## original name
        self.metric = "mrr"

        if meta_dict is None:
            self.dir_name = "_".join(name.split("-"))  ## replace hyphen with underline
            meta_dict = {"dir_name": self.dir_name}
        else:
            self.dir_name = meta_dict["dir_name"]
        assert name in data_max_timestep_map, f'{name} is not in data_max_timestep_map, keys: {data_max_timestep_map.keys()}'
        self.root = osp.join(root, self.dir_name, str(data_max_timestep_map[name]))
        self.meta_dict = meta_dict
        if "fname" not in self.meta_dict:
            self.meta_dict["fname"] = self.root + "/ml_" + self.name + ".csv"
            # todo: node feature
            nodefile = self.root + "/" + self.name + "_node_feat.csv"
            if osp.isfile(nodefile):
                self.meta_dict["nodefile"] = nodefile
            else:
                self.meta_dict["nodefile"] = None

        # initialize
        self._node_feat = None
        self._edge_feat = None
        self._full_data = None
        self._train_data = None
        self._val_data = None
        self._test_data = None

        if preprocess:
            self.pre_process()

        self.ns_sampler = NegativeEdgeSampler(
            dataset_name=self.name, strategy="hist_rnd"
        )
    
    # todo: node feature
    def pre_process(self):
        """
        Pre-process the dataset and generates the splits, must be run before dataset properties can be accessed
        generates the edge data and different train, val, test splits
        """
        # check if path to file is valid
        if not osp.exists(self.meta_dict["fname"]):
            raise FileNotFoundError(f"File not found at {self.meta_dict['fname']}")
        
        df = pd.read_csv(self.meta_dict["fname"])

        sources = df['u'].astype(np.longlong)
        destinations = df['i'].astype(np.longlong)
        timestamps = df['ts'].astype(np.float64)
        edge_idxs = df['idx'].astype(np.longlong)

        assert len(sources) == len(destinations) == len(timestamps) == len(edge_idxs), f'len not same: len(sources)  len(destinations)  len(timestamps)  len(edge_idxs): {len(sources)}  {len(destinations)}  {len(timestamps)}  {len(edge_idxs)}'

        # # DEBUG
        # node_id = 2
        # print(f'[{__file__}:{sys._getframe().f_lineno}] all neighbors of {node_id}: (v, t)')
        # neighbor_df = df[df['u']==node_id]
        # print(neighbor_df)
        # neighbor_df = df[df['i']==node_id]
        # print(neighbor_df)

        edge_label = np.ones(len(df))  # should be 1 for all pos edges
        edge_feat = np.zeros(len(df))  

        # write timestep_interval
        timesteps = df['timestamp']
        timestep2minmaxts = {}
        timestep_interval = [] # [timestep_interval[i], timestep_interval[i+1]) is the interval of ts for <|timei|>
        max_t_step = -1
        for i, ts in enumerate(timestamps):
            t_step = timesteps[i]
            if t_step > max_t_step:
                max_t_step = t_step
            if t_step not in timestep2minmaxts:
                timestep2minmaxts[t_step] = [-1, -1]
            min_ts = timestep2minmaxts[t_step][0]
            if min_ts<0 or ts<min_ts:
                timestep2minmaxts[t_step][0] = ts
            max_ts = timestep2minmaxts[t_step][1]
            if max_ts<0 or ts>max_ts:
                timestep2minmaxts[t_step][1] = ts
        assert max_t_step>=0, f'At least one timestep should exist! Check the csv file: {self.meta_dict["fname"]}'

        assert max_t_step == data_max_timestep_map[self.name], f'max_t_step={max_t_step} should be {data_max_timestep_map[self.name]}'
        num_timestep = max_t_step+1
        print(f'num_timestep = {num_timestep}')
        ts = 0
        timestep_interval.append(ts)
        for t_step in range(num_timestep):
            assert t_step in timestep2minmaxts, f"timestep {t_step} should in timestep2minmaxts (whose keys are {timestep2minmaxts.keys()}) given num_timestep={num_timestep}"
            [min_ts, max_ts] = timestep2minmaxts[t_step]

            assert min_ts>=ts, f'Min timestamp in future timestep should be greater than the timestamp bound (exclusive) of historical timesteps.\nmin_ts={min_ts} for t_step={t_step}, while the previous bound (exclusive) ={ts}'
            ts=max_ts+1
            timestep_interval.append(ts)

        # full_data = {
        #     "sources": sources,
        #     "destinations": destinations,
        #     "timestamps": timestamps,
        #     "edge_idxs": edge_idxs,
        #     "edge_label": edge_label,
        #     "timestep_interval": timestep_interval,
        # }
        full_data = {
            "sources": np.array(sources),
            "destinations": np.array(destinations),
            "timestamps": np.array(timestamps),
            "edge_idxs": np.array(edge_idxs),
            "edge_label": np.array(edge_label),
            "edge_feat": np.array(edge_feat),
            "timestep_interval": np.array(timestep_interval),
        }
        self._full_data = full_data

        test_ratio = val_ratio = 1/num_timestep # use the last (second last) timestep as test (val) set
        _train_mask, _val_mask, _test_mask = self.generate_splits(full_data, test_ratio=test_ratio, val_ratio=val_ratio)
        # self._train_mask = _train_mask
        # self._val_mask = _val_mask
        # self._test_mask = _test_mask
        self._train_mask = np.array(_train_mask)
        self._val_mask = np.array(_val_mask)
        self._test_mask = np.array(_test_mask)


class SimpleDyGEvaluator(Evaluator):
    def __init__(self, name: str, k_value: int = 10):
        r"""
        Parameters:
            name: name of the dataset
            k_value: the desired 'k' value for calculating metric@k
        """
        self.name = name
        self.k_value = k_value  # for computing `hits@k`
        self.valid_metric_list = ['hits@', 'mrr']

def get_link_prediction_simpledyg_data(dataset_name: str, num_timestep: int):
    """
    generate simpledyg data for link prediction task
    :param dataset_name: str, dataset name
    :return: node_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object), eval_neg_edge_sampler, eval_metric_name
            timestep_interval: [timestep_interval[i], timestep_interval[i+1]) is the interval of timestamps for <|timei|>
    """
    # Load data and train val test split
    dataset = SimpleDyGLinkPropPredDataset(name=dataset_name, root="datasets", preprocess=True)
    data = dataset.full_data

    src_node_ids = data['sources'].astype(np.longlong)
    dst_node_ids = data['destinations'].astype(np.longlong)
    node_interact_times = data['timestamps'].astype(np.float64)
    edge_ids = data['edge_idxs'].astype(np.longlong)
    labels = data['edge_label']

    num_edges = src_node_ids.shape[0]
    assert num_edges == data_num_edges_map[dataset_name], f'Number of edges are not matched! read edge num: {num_edges}, should be {data_num_edges_map[dataset_name]}'

    # union to get node set
    num_nodes = len(set(src_node_ids) | set(dst_node_ids))
    assert num_nodes == data_num_nodes_map[dataset_name], f'Number of nodes are not matched! read node num: {num_nodes}, should be {data_num_nodes_map[dataset_name]}'

    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    eval_neg_edge_sampler = dataset.negative_sampler
    dataset.load_val_ns()
    dataset.load_test_ns()
    eval_metric_name = dataset.eval_metric
    print(f'eval_metric_name: {eval_metric_name}')

    # # DEBUG
    # df = pd.DataFrame({'u': src_node_ids, 'i': dst_node_ids, 'ts': node_interact_times})
    # node_id = 2
    # print(f'[{__file__}:{sys._getframe().f_lineno}] all neighbors of {node_id}: (v, t)')
    # neighbor_df = df[df['u']==node_id]
    # print(neighbor_df)
    # neighbor_df = df[df['i']==node_id]
    # print(neighbor_df)

    if 'node_feat' not in data.keys():
        node_raw_features = np.zeros((num_nodes + 1, 1))
    else:
        node_raw_features = data['node_feat'].astype(np.float64)
        # deal with node features whose shape has only one dimension
        if len(node_raw_features.shape) == 1:
            node_raw_features = node_raw_features[:, np.newaxis]

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask], edge_ids=edge_ids[train_mask], labels=labels[train_mask])
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(test_data.num_interactions, test_data.num_unique_nodes))

    timestep_interval = data["timestep_interval"]
    print(f'timestep_interval: [timestep_interval[i], timestep_interval[i+1]) is the interval of timestamps for <|timei|>: \n{timestep_interval}')

    return node_raw_features, full_data, train_data, val_data, test_data, eval_neg_edge_sampler, eval_metric_name, timestep_interval

def load_simpledyg_test_data_in_simpledyg_format(dataset: str, timestamp: str) -> tuple[list, list]:
    # test data
    file_path = osp.join('resources', dataset, timestamp, 'test.link_prediction')
    # gt: ground truth
    file_path_gt = osp.join('resources', dataset, timestamp, 'test_gt.link_prediction')

    with open(file_path, encoding="utf-8") as f:
        data = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    with open(file_path_gt, encoding="utf-8") as f:
        data_gt = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    assert len(data) == len(data_gt)

    return data, data_gt

def simpledyg_test_data_to_standard_test_data_old(data: list, data_gt: list):
    '''
    # ! 此函数生成的数据有问题：
    # 因为用timestep作为时间戳，则每个顶点有很多邻居的时间戳是相同的，则我们获取u在ts之前的所有邻居时，在ts的邻居不会被获取，
    # 也就是说当我们预测某timestep时，只会取到该timestep之前的时间步中的邻居，这样就和simpledyg原先的场景不一样（原先的场景是一步步预测下去，
    # 当我们在预测<|time12|>中第二个邻居的时候，<|time12|>中的第一个邻居是要作为输入的）
    # ! 而如果我们通过给每个timestep中的邻居赋严格单调递增的timestamps时，有点麻烦：
    # 由于全图所有顶点的时间步是统一的，需要先获取每个时间步中的所有边，
    # 并记住需要保持的相对关系（一个顶点在一个时间步内的邻居的时间戳是递增的），然后再给它们赋时间戳，
    # 注意这是无向图，(u,v,t)和(v,u,t)是同一条边，因此还需要特殊考虑每个时间步中端点相同的边：直接可以想到的逻辑是，先搜集，然后同一条边仅保留一条（这种移除并不会破坏之前赋予的顺序，只是时间戳不是连续的罢了）
    # ! 但是注意，t是不知道的，或者说我们前面预赋予的那个并不代表真正的t，
    # 所以上面这种逻辑不够，需要对齐同一个时间步中不同顶点的(u,v)边，从而才能判断哪些边是重复的

    # 另外，eval_neg_edge_sampler也还没有写

    undirected graph
    data: rows in test.link_prediction
        e.g., <|endoftext|> <|history|> 112 <|time0|> 113 114 <|time1|> 118 <|time2|> 183 115 116 <|time3|> 118 <|endofhistory|>
    data_gt: rows in test_gt.link_prediction
        e.g., <|pre|> <|time12|> 169 688 <|endofpre|> <|endoftext|>

    treat edges in data_gt as test edges,
    edges in data and data_gt as the whole graph
    use the `x` in `<|timex|>` as interaction timestamps

    @return: same format as tgb test datasets
        full_data, test_data, eval_neg_edge_sampler, eval_metric_name
    '''

    edges = []

    src_pos_in_data_line = 2
    data_line_begin_inclusive = 3
    data_line_end_exclusive = -1
    data_gt_line_begin_inclusive = 1
    data_gt_line_end_exclusive = -2

    def line_to_edges(edges: list, lst: list, src: int):
        ts = -1
        for wd in lst:
            if SimpleDyGStandardize.is_time_token_str(wd):
                ts = SimpleDyGStandardize.time_token_to_num(wd)
                continue
            dst = int(wd)
            assert ts >= 0, f"Timestamp must be non negative. Wrong format in lst: {lst}"
            if (dst, src, ts) not in edges: # undirected graph
                edges.append((src, dst, ts))

    # collect edges
    for line, gt_line in zip(data, data_gt):
        lst = line.strip().split()
        src = lst[src_pos_in_data_line]
        line_to_edges(edges, lst[data_line_begin_inclusive: data_line_end_exclusive], src)

        lst = gt_line.strip().split()
        line_to_edges(edges, lst[data_gt_line_begin_inclusive: data_gt_line_end_exclusive], src)

        # # DEBUG
        # print(line)
        # print(gt_line)
        # break

    # sort according to ts
    sorted_edges = sorted(edges, key=lambda x: x[2])
    test_ts = sorted_edges[-1][2]

    src_node_ids = np.array([x[0] for x in sorted_edges], np.longlong)
    dst_node_ids = np.array([x[1] for x in sorted_edges], np.longlong)
    node_interact_times = np.array([x[2] for x in sorted_edges], np.float64)
    edge_ids = np.zeros(len(sorted_edges), np.longlong)
    labels = np.zeros(len(sorted_edges))

    test_mask = node_interact_times >= test_ts

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(test_data.num_interactions, test_data.num_unique_nodes))

    eval_neg_edge_sampler = None
    eval_metric_name = "mrr"
    return full_data, test_data, eval_neg_edge_sampler, eval_metric_name

def get_link_prediction_simpledyg_data_old(dataset: str, timestamp: str):
    data, data_gt = load_simpledyg_test_data_in_simpledyg_format(dataset, timestamp)
    return simpledyg_test_data_to_standard_test_data_old(data, data_gt)