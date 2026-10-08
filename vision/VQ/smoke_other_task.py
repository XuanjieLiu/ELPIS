"""Check subtraction training and evaluation without retaining experiment outputs."""
import os
from itertools import islice
from tempfile import TemporaryDirectory

import numpy as np
import torch

from loss_counter import LossCounter


def smoke_other_task_step(task, run_name):
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for this smoke check.')

    previous_dir = os.getcwd()
    with TemporaryDirectory(prefix='elpis-minus-smoke-') as work_dir:
        try:
            os.chdir(work_dir)
            task.resume()
            task.simple_fc.train()
            before = {name: p.detach().clone() for name, p in task.simple_fc.named_parameters()}
            optimizer = torch.optim.Adam(task.simple_fc.parameters(), lr=task.learning_rate)
            train_counter = LossCounter(['loss_z', 'accu', 'loss_recon'])
            task.one_epoch(0, train_counter, islice(task.train_loader, 1), False, optimizer)
            torch.cuda.synchronize()
            if len(train_counter.values_list) != 1 or not np.isfinite(train_counter.values_list).all():
                raise RuntimeError('Expected one training minibatch with finite losses and accuracy.')
            for name, p in task.simple_fc.named_parameters():
                if not torch.isfinite(p).all() or p.grad is None or not torch.isfinite(p.grad).all():
                    raise RuntimeError(f'Invalid parameters or gradients: {name}')
            for head in ('classify_fc_net', 'recon_fc_net'):
                if not any(not torch.equal(before[name], p) for name, p in task.simple_fc.named_parameters()
                           if name.startswith(head + '.')):
                    raise RuntimeError(f'The optimizer did not update {head}.')

            task.simple_fc.eval()
            eval_counter = LossCounter(['loss_z', 'accu', 'loss_recon'])
            with torch.no_grad():
                task.one_epoch(0, eval_counter, islice(task.eval_loader, 1), False, None)
            if len(eval_counter.values_list) != 1 or not np.isfinite(eval_counter.values_list).all():
                raise RuntimeError('Expected one evaluation minibatch with finite losses and accuracy.')
            print(f'MINUS SMOKE PASS: {run_name}; task={task.other_task_config["task_name"]}; '
                  f'optimizer_steps=1; train={train_counter.values_list[0]}; '
                  f'eval={eval_counter.values_list[0]}', flush=True)
        finally:
            os.chdir(previous_dir)
            task.delete_loader_iter()
