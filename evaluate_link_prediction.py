import logging
import time

import json
import os
from tqdm import tqdm
import numpy as np

from utils.gpt2_args_parser import ArgsParser

from adapt_problem import SimpleDyGStandardize, evaluate_model_link_prediction
from adapt_dataset import get_link_prediction_simpledyg_data, load_simpledyg_test_data_in_simpledyg_format, SimpleDyGEvaluator
from utils.utils import get_neighbor_sampler, set_random_seed
from utils.DataLoader import get_idx_data_loader
 

def standard_test(args):
    # set some args
    args.sample_neighbor_strategy = 'recent'
    args.time_scaling_factor = 1e-6
    args.dataset_name = args.dataset
    args.model_name = 'SimpleDyG'
    args.batch_size = args.per_gpu_train_batch_size

    # get data for training, validation and testing
    node_raw_features, full_data, train_data, val_data, test_data, eval_neg_edge_sampler, eval_metric_name, timestep_interval = \
        get_link_prediction_simpledyg_data(dataset_name=args.dataset_name, num_timestep=int(args.timestamp))

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)
    
    
    # # DEBUG
    # model = SimpleDyGStandardize(args.output_dir, full_neighbor_sampler, timestep_interval) # load model
    # SimpleDyGStandardize.init_tokens(int(args.timestamp), model.tokenizer)
    # # SimpleDyGStandardize.print_special_tokens(model.tokenizer)
    # model.eval()
    # model.to('cuda')
    # src = 2
    # dsts = [1680]
    # ts = 10901477
    # dst_probas = model.predict_link_probas(src, dsts, ts)
    # print(f'predicted probas of dsts{dsts}: {dst_probas[0]}')
    # return

    # get data loaders
    # since the version 2 of tgbl-wiki has included all possible negative destinations for each positive edge, we set batch size to 1 to reduce the memory cost
    if args.dataset_name == "tgbl-wiki":
        val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=1, shuffle=False)
        test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=1, shuffle=False)
    else:
        val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
        test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, test_metric_all_runs = [], []

    for run in range(args.num_runs):
        set_random_seed(seed = args.seed)

        args.load_model_name = f'{args.model_name}_seed{args.seed}'
        args.save_result_name = f'eval_{args.load_model_name}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # load model
        model = SimpleDyGStandardize(args.output_dir, full_neighbor_sampler, timestep_interval) 

        # init special tokens
        SimpleDyGStandardize.init_tokens(int(args.timestamp), model.tokenizer)
        # SimpleDyGStandardize.print_special_tokens(model.tokenizer)

        # model.eval()
        model.to('cuda')

        evaluator = SimpleDyGEvaluator(name=args.dataset_name)

        val_metrics = evaluate_model_link_prediction(model=model,
                                                    neighbor_sampler=full_neighbor_sampler,
                                                    evaluate_idx_data_loader=val_idx_data_loader,
                                                    evaluate_neg_edge_sampler=eval_neg_edge_sampler,
                                                    evaluate_data=val_data,
                                                    eval_stage='val',
                                                    eval_metric_name=eval_metric_name,
                                                    evaluator=evaluator)

        test_metrics = evaluate_model_link_prediction(model=model,
                                                    neighbor_sampler=full_neighbor_sampler,
                                                    evaluate_idx_data_loader=test_idx_data_loader,
                                                    evaluate_neg_edge_sampler=eval_neg_edge_sampler,
                                                    evaluate_data=test_data,
                                                    eval_stage='test',
                                                    eval_metric_name=eval_metric_name,
                                                    evaluator=evaluator)

        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        for metric_name in val_metrics[0].keys():
            average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
            logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
            val_metric_dict[metric_name] = average_val_metric

        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in val_metric_all_runs[0].keys():
        logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
        logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                    f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    
def simpledyg_fashion_test(args):
    model = SimpleDyGStandardize(args.output_dir) # load model
    SimpleDyGStandardize.init_tokens(int(args.timestamp), model.tokenizer)
    SimpleDyGStandardize.print_special_tokens(model.tokenizer)

    data, data_gt = load_simpledyg_test_data_in_simpledyg_format(args.dataset, args.timestamp)

    model.eval()
    model.to('cuda')

    for i, (input_text, text_gt) in enumerate(tqdm(zip(data, data_gt))):
        # DEBUG
        if i:
            break

        print(f'simpledyg fashion input: {input_text}')
        print(f'simpledyg fashion ground truth: {text_gt}')
        probas = model.predict_next_neighbor(input_text)

if __name__ == "__main__":
    args = ArgsParser().parse()

    # simpledyg_fashion_test(args)

    standard_test(args)

    

