import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.linkproppred.evaluate import Evaluator as LinkPredictionEvaluator
from utils.DataLoader import Data
from utils.utils import NeighborSampler

class SimpleDyGStandardize:
    '''
    class member
    '''
    SPL_TOKENS_STR = ['<|history|>','<|endofhistory|>','<|pre|>','<|endofpre|>','<|endoftext|>','[PAD]'] # will be added with <timex> in main
    SPL_TOKENS = []
    STOP_SPL_TOKENS_STR = ["<|endoftext|>"]
    STOP_SPL_TOKENS = []
    
    def init_tokens(timestamp: int, tokenizer: PreTrainedTokenizerFast):
        SimpleDyGStandardize.SPL_TOKENS_STR += ['<|time'+str(i)+'|>' for i in range(timestamp + 1)]   

        for wd in SimpleDyGStandardize.STOP_SPL_TOKENS_STR:
            SimpleDyGStandardize.STOP_SPL_TOKENS.append(tokenizer.encode(wd)) # Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.
        for wd in SimpleDyGStandardize.SPL_TOKENS_STR:
            SimpleDyGStandardize.SPL_TOKENS.append(tokenizer.encode(wd)) # Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

    def print_decode_tokens(tokens: list, tokenizer: PreTrainedTokenizerFast):
        res = []
        for token in tokens:
            res.append(tokenizer.decode(token))
        print(res)

    def print_special_tokens(tokenizer: PreTrainedTokenizerFast):
        print(f'SPL_TOKENS: {SimpleDyGStandardize.SPL_TOKENS}')
        SimpleDyGStandardize.print_decode_tokens(SimpleDyGStandardize.SPL_TOKENS, tokenizer)
        print(f'STOP_SPL_TOKENS: {SimpleDyGStandardize.STOP_SPL_TOKENS}')
        SimpleDyGStandardize.print_decode_tokens(SimpleDyGStandardize.STOP_SPL_TOKENS, tokenizer)

    def is_time_token_str(wd: str):
        if (wd in SimpleDyGStandardize.SPL_TOKENS_STR) and ('time' in wd):
            return True
        return False
    
    def time_token_to_num(wd: str):
        # <|time0|>
        wd = wd.strip('<>|')[len('time'):]
        return int(wd)

    '''
    instance member
    '''
    def __init__(self, checkpoint_path: str, neighbor_sampler: NeighborSampler, timestep_interval: list):
        '''
        load model
        '''

        model_checkpoint = os.path.join(checkpoint_path, 'checkpoint-0')
        print(f'load model from {model_checkpoint}')
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_checkpoint,'tokenizer.json'), use_fast=False)
        self.model = GPT2LMHeadModel.from_pretrained(model_checkpoint)

        self.neighbor_sampler = neighbor_sampler # fetch the graph
        self.timestep_interval = timestep_interval # timestep_interval: [timestep_interval[i], timestep_interval[i+1]) is the interval of timestamps for <|timei|>

    def eval(self):
        self.model.eval()
    
    def to(self, device: str):
        self.model.to(device)    

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

    def temporal_one_hop_neighbors_to_input_text(self, u: int, ts: int) -> str:
        '''
        encode the neighbor sequence

        @param:
        u: node
        ts: timestamp, fetch the neighbors before (<) ts
        timestep_interval: [timestep_interval[i], timestep_interval[i+1]) is the interval of timestamps for <|timei|>

        @return: neighbor sequences with special_tokens of u
        <|endoftext|> <|history|> u <|time0|> 2 ... <|endofhistory|>
        '''

        sep = ' '
        res = f'<|endoftext|>{sep}<|history|>{sep}{u}'
        neighbors, _, timestamps, _ = self.neighbor_sampler.find_neighbors_before(u, ts)
        curr_t_step = -1
        for nbr, t in zip(neighbors, timestamps):
            # return index i, which satisfies 
            # left: a[i-1] < v <= a[i]
            # right: a[i-1] <= v < a[i]
            t_step = np.searchsorted(self.timestep_interval, t, side='right')
            assert (t_step and t_step<len(self.timestep_interval)), f'Fail to locate the time step where ts={t} in! timestep_interval: {self.timestep_interval}'
            t_step -= 1
            if t_step != curr_t_step:
                curr_t_step = t_step
                res += f'{sep}<|time{curr_t_step}|>'
            res += f'{sep}{nbr}'
        res += f'{sep}<|endofhistory|>'
        return res

    def predict_next_neighbor(self, input_text: str) -> torch.FloatTensor:
        '''
        predict the next neighbor
        return the softmax probability
        '''

        model_context_max_len = self.model.config.n_ctx
        
        # Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary
        indexed_tokens = self.tokenizer.encode(input_text) 
        input_len = len(indexed_tokens)
        
        if len(indexed_tokens) > model_context_max_len:
            print(f'Warn: len(indexed_tokens)={len(indexed_tokens)} > model_context_max_len={model_context_max_len}: Context len exceeds.')
            # indexed_tokens = indexed_tokens[-1000:]
            indexed_tokens = indexed_tokens[-model_context_max_len:]
            print(f'Will only keep the last model_context_max_len tokens of indexed_tokens, which is\n{indexed_tokens}')

        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to('cuda')

        fail_to_predict = False
        probas = None
        while True:
            # out of context len, then ends the prediction
            if len(indexed_tokens) >= model_context_max_len-len(SimpleDyGStandardize.SPL_TOKENS_STR):
                fail_to_predict = True
                break

            outputs = self.model(tokens_tensor) # predict the next token of tokens_tensor (format: probability of each token in the vocabulary)
            predictions = outputs[0]
            probas = predictions[0, -1, :]
            predicted_index = torch.argmax(probas).item() # the predicted token, before decoding

            indexed_tokens += [predicted_index] 

            # if predicted_index is stop token, then ends the prediction
            if [predicted_index] in SimpleDyGStandardize.STOP_SPL_TOKENS:
                fail_to_predict = True
                break
            # if predicted_index is special token, continue to predict
            # else, we have abtained the prediction
            if [predicted_index] not in SimpleDyGStandardize.SPL_TOKENS:
                break

            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')

        # DEBUG
        print('Predicted tokens: ', end='')
        print(indexed_tokens[input_len:])
        print('Predicted text: ', end='')
        predicted_text = self.tokenizer.decode(indexed_tokens)
        predicted_list = (predicted_text.split())[input_len:] 
        print(' '.join(predicted_list))
        
        if fail_to_predict:
            print('Fail to predict the next neighbor. Currently the predicted text is:')
            predicted_text = self.tokenizer.decode(indexed_tokens)
            predicted_list = (predicted_text.split())[input_len:] 
            print(' '.join(predicted_list))
            return None
        else:
            # # DEBUG
            # print(f'probas before softmax: {probas}')
            softmax_func = torch.nn.Softmax(dim=0)
            probas = softmax_func(probas)
            # print(f'probas after softmax: {probas}')
            # max_idx = torch.argmax(probas).item()
            # print(f'max proba: {probas[max_idx]}')

            return probas
    
    def predict_link_probas(self, src: int, dsts: list, ts: int) -> list:
        '''
        predict the probability of (src, dst in dsts, ts)
        '''

        src_nbr_text = self.temporal_one_hop_neighbors_to_input_text(src, ts)
        # print(f'convert neighbors of {src} to simpledyg fashion text: {src_nbr_text}')

        # probas of src's next neighbor being each node
        # note that the idx on probas are tokenizer.encode(str(node_id))
        probas = self.predict_next_neighbor(src_nbr_text) 
        if probas is None:
            print(f'Fail to predict the next neighbor for node {src}')
            return []

        # # DEBUG
        # min_dst_proba = -1
        # print('predicted proba for dsts:')

        dst_probas = []
        for dst in dsts:
            token = self.tokenizer.encode(str(dst))
            p = probas[token][0].detach().item() 
            dst_probas.append(p)

            # # DEBUG
            # if min_dst_proba<0 or p<min_dst_proba:
            #     min_dst_proba = p
            # print(f'node: {dst} proba: {p}')

        # # DEBUG
        # print('nodes that predicted to be more likely to be neighbors than some dsts:')
        # for token, p in enumerate(probas):
        #     if p>min_dst_proba:
        #         print(f'node: {self.tokenizer.decode(token)} proba: {p}')

        return dst_probas


def evaluate_model_link_prediction(model: SimpleDyGStandardize, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, eval_stage: str,
                                   eval_metric_name: str, evaluator: LinkPredictionEvaluator):
    """
    evaluate models on the link prediction task
    :param model: SimpleDyGStandardize, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler from TGB
    :param evaluate_data: Data, data to be evaluated
    :param eval_stage: str, specifies the evaluate stage to generate negative edges, can be 'val' or 'test'
    :param eval_metric_name: str, name of the evaluation metric
    :param evaluator: LinkPredictionEvaluator, link prediction evaluator
    :return:
    """
    model.set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_metrics = []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            # batch_neg_dst_node_ids_list: a list of list, where each internal list contains the ids of negative destination nodes for a positive source node
            # contain batch lists, each list with length num_negative_samples_per_node (20 in the TGB evaluation)
            batch_neg_dst_node_ids_list = evaluate_neg_edge_sampler.query_batch(pos_src=batch_src_node_ids, pos_dst=batch_dst_node_ids,
                                                                                pos_timestamp=batch_node_interact_times, split_mode=eval_stage)
            
            # for each sample
            for i in range(len(batch_src_node_ids)):
                probas = model.predict_link_probas(src=batch_src_node_ids[i], dsts=[batch_dst_node_ids[i]] + batch_neg_dst_node_ids_list[i], ts=batch_node_interact_times[i])
                
                # compute metric
                input_dict = {
                    # use slices instead of index to keep the dimension of y_pred_pos
                    "y_pred_pos": np.array(probas[0: 1]),
                    "y_pred_neg": np.array(probas[1:]),
                    "eval_metric": [eval_metric_name],
                }
                evaluate_metrics.append({eval_metric_name: evaluator.eval(input_dict)[eval_metric_name]})

            evaluate_idx_data_loader_tqdm.set_description(f'{eval_stage} for the {batch_idx + 1}-th batch')

    return evaluate_metrics


