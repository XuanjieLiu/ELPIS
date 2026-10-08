import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from .load_batch_record import ExpGroup
from .multi_key_compare import MultiKeyCompareGroup
import numpy as np
from typing import List
from loss_counter import LossCounter, RECORD_PATH_DEFAULT


EXP_NAME_LIST = [
    # "multistyle_elpis",
    # "multistyle_no_symmetry",
    # "multistyle_full_data",
    # "multistyle_recon_only",

    # "2025.07.02_20vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_SingleStyleMahjong_nothing",
    # "2025.07.02_20vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_SingleStyleMahjong_PureVQ",
    # "2025.07.02_20vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_SingleStyleMahjong_symm",
    # "2025.07.02_20vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_SingleStyleMahjong_trainAll",

    # "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_nothing",
    # "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_PureVQ",
    # "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_symm",
    # "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_trainAll",

    # "dots_elpis",
    # "dots_no_symmetry",
    # "dots_full_data",
    # "dots_recon_only",

    "eastern_tally_elpis",
    "eastern_tally_no_symmetry",
    "eastern_tally_full_data",
    "eastern_tally_recon_only",

    "western_tally_elpis",
    "western_tally_no_symmetry",
    "western_tally_full_data",
    "western_tally_recon_only",

]
group_list = [
    ExpGroup(
        exp_name=name,
        exp_alias='N/A',
        sub_exp=[i for i in range(1, 21)],
        record_name="minus_16_1.1_eval_record.txt",
        is_load_record=False
    ) for name in EXP_NAME_LIST
]


EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../VQ/exp')
EXP_NUM_LIST = [str(i) for i in range(1, 21)]
OTHER_TASK_EXP_NUM_LIST = [str(i) for i in range(1, 11)]

COMPARE_KEYS = ['accu', 'loss_recon']
COMPARE_KEYS_NAME = ['Accuracy', 'Repr_pred_loss']
IS_MAX_BETTER = [True, False]
OUTPUT_PATH = "train_test_summary/"
EXTREME_NUM = 1
ITER_AFTER = 0
Y_NAME = "Plus Accuracy (max=1.0) ↑"


def exp_group2compare_group(exp_group: ExpGroup):
    y_list = []
    for record in exp_group.sub_record:
        y = []
        i = 0
        for key in COMPARE_KEYS:
            record_data = record[key]
            mean = float(np.mean(record_data.find_topMin_after_iter(EXTREME_NUM, ITER_AFTER, IS_MAX_BETTER[i])))
            y.append(mean)
            i += 1
        y_list.append(y)
    compare_group = MultiKeyCompareGroup(
        title=exp_group.exp_alias,
        y_name=Y_NAME,
        keys=COMPARE_KEYS_NAME,
        values=y_list
    )
    return compare_group


def gen_compare_groups(exp_groups: List[ExpGroup]):
    compare_groups = []
    for eg in exp_groups:
        compare_groups.append(exp_group2compare_group(eg))
    return compare_groups


def group_to_subgroups(eg: ExpGroup, sub_exp_num_list=OTHER_TASK_EXP_NUM_LIST):
    sub_eg_list = []
    for other_task in EXP_NUM_LIST:
        sub_eg = ExpGroup(
            exp_name=os.path.join(eg.exp_name, other_task),
            exp_alias=eg.exp_alias,
            sub_exp=sub_exp_num_list,
            record_name=eg.record_name,
        )
        sub_eg_list.append(sub_eg)
    return sub_eg_list


def find_all_results(eg: ExpGroup, sub_exp_num_list=OTHER_TASK_EXP_NUM_LIST):
    sub_eg_list = group_to_subgroups(eg, sub_exp_num_list=sub_exp_num_list)
    cg_list = gen_compare_groups(sub_eg_list)
    all_result = [[] for i in range(0, len(COMPARE_KEYS))]
    for i in range(0, len(cg_list)):
        for j in range(0, len(COMPARE_KEYS)):
            values = [item[j] for item in cg_list[i].values]
            all_result[j].extend(values)
    return all_result


def summary_an_exp(eg: ExpGroup):
    exp_dir = os.path.join(EXP_ROOT_PATH, eg.exp_name)
    print(f'Exp path: {exp_dir}')
    all_result = find_all_results(eg)
    print(f'All results for {eg.exp_alias}: {all_result}')
    results_str = []
    for j in range(0, len(COMPARE_KEYS)):
        values = all_result[j]
        mean = float(np.mean(values))
        std = float(np.std(values))
        result = f'{COMPARE_KEYS_NAME[j]}: {mean:.2f} \\pm {std:.2f}'
        results_str.append(result)
        print(result)
    log_result(exp_dir, results_str)


def log_result(exp_dir, results_str):
    result_path = os.path.join(exp_dir, 'other_task_summary_result.txt')
    with open(result_path, 'w') as f:
        for result in results_str:
            f.write(result + '\n')


if __name__ == '__main__':
    for eg in group_list:
        summary_an_exp(eg)
