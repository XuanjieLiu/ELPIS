import argparse
from pathlib import Path
import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from common_func import load_config_from_exp_name, EXP_ROOT, record_num_list, find_optimal_checkpoint_num_by_train_config
from experiment_registry import resolve_experiment_name
from loss_counter import read_record
from dataloader_plus import MultiImgDataset


EXP_NUM_LIST = [str(i) for i in range(1, 21)]
EXP_NAME_LIST = (Path(__file__).resolve().parents[1] / 'reproduction/fig2b_runs.txt').read_text().splitlines()
EVAL_RECORD_NAME = 'plus_eval.txt'

EVAL_KEYS = ['eval_emb_select_accu', 'eval_emb_select_accu_cycle', 'eval_one2n_accu',
             'eval_one2n_accu_cycle', 'eval_one2one_accu', 'eval_one2one_accu_cycle']

# EVAL_KEYS = ['train_accu', 'eval_accu', 'eval_accu_2']


def batch_statistic(exp_names=None):
    is_summary_two_eval = 'eval_accu_2' in EVAL_KEYS and 'eval_accu' in EVAL_KEYS
    if is_summary_two_eval:
        EVAL_KEYS.append('eval_accu_all')
    for name in (EXP_NAME_LIST if exp_names is None else exp_names):
        exp_name = resolve_experiment_name(name)
        record_lists = [[] for _ in range(len(EVAL_KEYS))]
        exp_path = os.path.join(EXP_ROOT, exp_name)
        config = load_config_from_exp_name(exp_name)
        for sub_exp in EXP_NUM_LIST:
            sub_exp_path = os.path.join(exp_path, sub_exp)
            optimal_checkpoint_num = find_optimal_checkpoint_num_by_train_config(sub_exp_path, config)
            eval_txt_path = os.path.join(sub_exp_path, EVAL_RECORD_NAME)
            eval_record = read_record(eval_txt_path)
            for i in range(len(EVAL_KEYS)):
                key = EVAL_KEYS[i]
                if key in eval_record.keys():
                    value = eval_record[key].filter_Y_by_X_nums([optimal_checkpoint_num])[1][0]
                    record_lists[i].append(value)
        if is_summary_two_eval:
            eval_size_1 = len(MultiImgDataset(config['plus_eval_set_path']))
            eval_size_2 = len(MultiImgDataset(config['plus_eval_set_path_2']))
            idx_1 = EVAL_KEYS.index('eval_accu')
            idx_2 = EVAL_KEYS.index('eval_accu_2')
            idx_accu_all = EVAL_KEYS.index('eval_accu_all')
            for i in range(len(record_lists[idx_1])):
                accu_all = ((record_lists[idx_1][i] * eval_size_1 + record_lists[idx_2][i] * eval_size_2) /
                            (eval_size_1 + eval_size_2))
                record_lists[idx_accu_all].append(accu_all)

        for i in range(len(EVAL_KEYS)):
            key = EVAL_KEYS[i]
            results_path = os.path.join(exp_path, f'statistic_{key}.txt')
            record_num_list(results_path, record_lists[i], EXP_NUM_LIST)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize Fig. 2B training-ratio experiments.')
    parser.add_argument('run_names', nargs='*', help='Default: all 18 Fig. 2B groups.')
    args = parser.parse_args()
    batch_statistic(args.run_names or None)

