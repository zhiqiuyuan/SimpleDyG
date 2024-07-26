import os

# datasets = ['UCI_13', 'ML_10M_13', 'hepth', 'dialog']
datasets = ['ML_10M_13', 'hepth', 'dialog']

def dataset_name_to_train_sh(g):
    return f'train_seed_{g}.sh'
def dataset_name_to_test_sh(g):
    return f'test_seed_{g}.sh'

def train_sh_to_test_sh(datasets):
    for g in datasets:
        f = dataset_name_to_train_sh(g)
        outf = dataset_name_to_test_sh(g)

        res=''
        with open(f) as fin:
            for line in fin:
                if 'for Timestamp in' in line:
                    res += 'num_runs=1\n' + line
                elif 'python' in line:
                    res += line.replace('main.py', 'evaluate_link_prediction.py')
                elif 'run_seed' in line:
                    white_head = line.strip('\n').split('-')[0]
                    res += line.strip('\n') + ' \\\n' + white_head + '--num_runs=$num_runs\n'
                else:
                    res += line

        with open(outf, 'w') as fout:
            fout.write(res)

def train_all(datasets):
    for g in datasets:
        f = dataset_name_to_train_sh(g)
        os.system(f'bash {f}')
def test_all(datasets):
    for g in datasets:
        f = dataset_name_to_test_sh(g)
        os.system(f'bash {f}')

if __name__ == "__main__":
    # train_sh_to_test_sh(datasets)
    test_all(datasets)