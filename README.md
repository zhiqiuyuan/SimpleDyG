# SimpleDyG

The code and datasets used for our paper "On the Feasibility of Simple Transformer for Dynamic Graph Modeling" which is accepted by WWW 2024.

# Requirements and Installation 我装的环境是：modcard服务器上，SimpleDyG conda环境

```bash
conda create -n SimpleDyG python=3.9
```

python>=3.9

pytorch>=1.9.1

transformers>=4.24.0

The package can be installed by running the following command.

`pip install -r requirements.txt`

注意，其中的torch的版本，需要和你机器上cuda的版本匹配（可以先按requirements中的装了，运行的时候报错的时候去搜应该装什么版本的torch）
比如报错如下：
A100-SXM4-40GB with CUDA capability sm_80 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.
If you want to use the A100-SXM4-40GB GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
然后搜到的[解决方法](https://github.com/acids-ircam/RAVE/discussions/123)是装这个版本的
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

安装环境中的error：
```bash
ERROR: sqlalchemy 2.0.31 has requirement typing-extensions>=4.6.0, but you'll have typing-extensions 4.5.0 which is incompatible.
ERROR: pyfume 0.3.4 has requirement numpy==1.24.4, but you'll have numpy 1.24.3 which is incompatible.
ERROR: pyfume 0.3.4 has requirement pandas==1.5.3, but you'll have pandas 2.0.1 which is incompatible.
```


# Benchmark Datasets and Preprocessing (Optional)

You can download the raw datasets and run the preprocessing by yourself. 

Or you can run the training directly using the preprocessed data in `./resources`

## Raw data:

- UCI and  ML-10M: the raw data is the same with  https://github.com/aravindsankar28/DySAT

- Hepth: The dataset can be downloaded from the KDD cup:  https://www.cs.cornell.edu/projects/kddcup/datasets.html

- MMConv: we provide the raw data downloaded from https://github.com/liziliao/MMConv. It is a text-based multi-turn dialog dataset. We preprocess the data by representing the dialog as a graph for each turn based on the annotated attributes. We provide the preprocessed data in `all/data/dialog`

## Let's do preprocessing!  

All the datasets and preprocessing code are in folder `/all_data`. For each dataset, run:

`python preprocess.py ` 


The preprocessed data contains:

- `ml_dataname.csv`: the columns: *u*, *i* is the node Id. *ts* is the time point. *timestamp* is the **coarse-grained time steps** for **temporal alignment**.
所以划分大的time step，是在数据集划分为train/val/test之前，就把整个数据集划分为T个time step了，然后用最后一个time step做测试集，倒数第二个做验证集
- `ml_dataname.npy`: the raw link feature. 
- `ml_dataname_node.npy`: the raw node feature. 

Transfer the preprocessed data into sequences for the Transformer model: 

`bash csv2res.sh`

The final data is saved in  `./resources`, including the train/val/test data.

**We provide the final processed data for running our model.**

# Train the model 

`bash train_UCI_13.sh`

During training, the following output files are generated:

- `./tokenizers`: the tokenizers for each timestamp

- `./vocabs`: the vocab for each timestamp

- `./output`: the saved checkpoint of the model

- `./results`: the output results and metrics 

- `./runs`: the tensorboard results


# Evaluation 

`bash eval_single_step.py`

# Citation
```
@inproceedings{wu2024feasibility,
  title={On the Feasibility of Simple Transformer for Dynamic Graph Modeling},
  author={Wu, Yuxia and Fang, Yuan and Liao, Lizi},
  booktitle={Proceedings of the ACM on Web Conference 2024},
  pages={870--880},
  year={2024}
}

```
