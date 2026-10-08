import sys
import os
import argparse
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from train import is_need_train, PlusTrainer
from experiment_registry import load_experiment_config, resolve_experiment_name

parser = argparse.ArgumentParser(description='Train the selected experiments.')
parser.add_argument('run_names', nargs='+')
parser.add_argument('--smoke-test', action='store_true',
                    help='Run one optimizer step per run in a temporary directory, then clean up.')
args = parser.parse_args()

EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
sys.path.append(EXP_ROOT_PATH)
EXP_NAME_LIST = [resolve_experiment_name(name) for name in args.run_names]
# EXP_NUM_LIST = [str(i) for i in range(1, 21)]
if __name__ == '__main__':
    print(f'Experiment names: {EXP_NAME_LIST}')
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
        os.chdir(exp_path)
        print(f'Exp path: {exp_path}')
        config = load_experiment_config(exp_name)
        print(config)
        if args.smoke_test:
            from smoke_train import smoke_train_step
            smoke_train_step(config, exp_name)
            continue
        num_repeats = config.get('num_repeats', 20)
        init_random_split_seed = config.get('random_split_seed', None)
        print(f'Number of sub-experiments: {num_repeats}')
        exp_num_list = [str(i) for i in range(1, num_repeats + 1)]
        for exp_num in exp_num_list:
            os.makedirs(exp_num, exist_ok=True)
            sub_exp_path = os.path.join(exp_path, exp_num)
            if init_random_split_seed is not None:
                config['random_split_seed'] = init_random_split_seed + int(exp_num)
                print(f'Updated random_split_seed to {config["random_split_seed"]}')
            print(f'Sub-Exp path: {sub_exp_path}')
            os.chdir(sub_exp_path)
            if is_need_train(config):
                trainer = PlusTrainer(config)
                trainer.train()
            os.chdir(exp_path)
