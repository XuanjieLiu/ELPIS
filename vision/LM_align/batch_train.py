import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from train_align import AlignTrain
from experiment_registry import load_experiment_config, resolve_experiment_name

if len(sys.argv) < 2:
    print("Usage: python LM_align/batch_train.py icon_alignment")
    sys.exit()

EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
sys.path.append(EXP_ROOT_PATH)
EXP_NAME_LIST = [resolve_experiment_name(name, 'LM_align') for name in sys.argv[1:]]
# EXP_NUM_LIST = [str(i) for i in range(1, 21)]
if __name__ == '__main__':
    print(f'Experiment names: {EXP_NAME_LIST}')
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
        os.chdir(exp_path)
        print(f'Exp path: {exp_path}')
        config = load_experiment_config(exp_name, 'LM_align', 'config')
        print(config)
        num_repeats = config.get('num_repeats', 20)
        print(f'Number of sub-experiments: {num_repeats}')
        exp_num_list = [str(i) for i in range(1, num_repeats + 1)]
        for exp_num in exp_num_list:
            os.makedirs(exp_num, exist_ok=True)
            sub_exp_path = os.path.join(exp_path, exp_num)
            print(f'Sub-Exp path: {sub_exp_path}')
            os.chdir(sub_exp_path)
            config['sub_exp_id'] = exp_num
            trainer = AlignTrain(config)
            trainer.train()
            os.chdir(exp_path)
