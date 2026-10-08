"""One real training minibatch, without changing experiment configurations."""
import os
from itertools import islice
from tempfile import TemporaryDirectory

import numpy as np
import torch

from loss_counter import LossCounter
from train import PlusTrainer
from visual_imgs import VisImgs


def smoke_train_step(config, run_name):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Run smoke tests inside an srun job step.')
    if not torch.cuda.is_available():
        raise RuntimeError('The allocated GPU is not visible inside this job step.')

    previous_dir = os.getcwd()
    with TemporaryDirectory(prefix='elpis-smoke-') as work_dir:
        try:
            os.chdir(work_dir)
            trainer = PlusTrainer(config)
            trainer.model.train()
            parameters = [p for p in trainer.model.parameters() if p.requires_grad]
            before = [p.detach().clone() for p in parameters]
            optimizer = torch.optim.Adam(parameters, lr=trainer.learning_rate)
            counter = LossCounter([])
            # Use the same data loader, loss, backward pass and optimizer as training.
            trainer.one_epoch(0, counter, islice(trainer.train_loader, 1),
                              False, VisImgs(), optimizer)
            torch.cuda.synchronize()
            if len(counter.values_list) != 1:
                raise RuntimeError('Expected exactly one training minibatch.')
            if not np.isfinite(counter.values_list).all():
                raise RuntimeError('Non-finite training losses.')
            if not all(torch.isfinite(p).all().item() for p in parameters):
                raise RuntimeError('Non-finite model parameters after the step.')
            if not any(not torch.equal(old, p) for old, p in zip(before, parameters)):
                raise RuntimeError('The optimizer did not update any model parameters.')
            print(f'SMOKE PASS: {run_name}; optimizer_steps=1; '
                  f'losses={counter.values_list[0]}', flush=True)
        finally:
            os.chdir(previous_dir)
