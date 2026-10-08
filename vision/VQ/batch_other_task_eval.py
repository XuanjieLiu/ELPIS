import sys
import os
import argparse
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from other_task_eval import OtherTask, is_need_train
from common_func import find_optimal_checkpoint_num_by_train_config, EXP_ROOT
from experiment_registry import load_experiment_config, resolve_experiment_name
import torch.multiprocessing as mp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train subtraction tasks for selected experiments.')
    parser.add_argument('run_names', nargs='+')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Use repetition 1; run one training and one evaluation minibatch per task, then clean up.')
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    sys.path.append(EXP_ROOT)
    EXP_NAME_LIST = [resolve_experiment_name(name) for name in args.run_names]
    EXP_NUM_LIST = ['1'] if args.smoke_test else [str(i) for i in range(1, 21)]

    # Batch eval minus

    for exp_num in EXP_NUM_LIST:
        for exp_name in EXP_NAME_LIST:
            exp_path = os.path.join(EXP_ROOT, exp_name)
            os.chdir(exp_path)
            print(f'Exp path: {exp_path}')
            pretrained_config = load_experiment_config(exp_name)
            other_task_config = load_experiment_config(exp_name, config_name='other_tasks_config')
            print(pretrained_config)
            print(other_task_config)
            sub_exp_path = os.path.join(exp_path, exp_num)
            for i in range(0, len(other_task_config)):
                print(f'Sub-Exp path: {sub_exp_path}')
                os.chdir(sub_exp_path)
                config = other_task_config[i]
                optimal_checkpoint_finding_config = config.get('optimal_checkpoint_finding_config', None)
                optimal_check_point = find_optimal_checkpoint_num_by_train_config(sub_exp_path, pretrained_config, optimal_checkpoint_finding_config)
                config['pretrained_path'] = f'checkpoint_{optimal_check_point}.pt'
                print(f'Optimal checkpoint: {optimal_check_point}')
                other_task = OtherTask(pretrained_config, config)
                if args.smoke_test:
                    from smoke_other_task import smoke_other_task_step
                    smoke_other_task_step(other_task, exp_name)
                    continue
                num_heads_per_model = config.get('num_heads_per_model', 20)
                other_task_exp_num_list = [str(i) for i in range(1, num_heads_per_model + 1)]
                for j in other_task_exp_num_list:
                    sub_sub_exp_path = os.path.join(sub_exp_path, j)
                    os.makedirs(sub_sub_exp_path, exist_ok=True)
                    os.chdir(sub_sub_exp_path)
                    if is_need_train(config):
                        other_task.train()
                other_task.delete_loader_iter()
