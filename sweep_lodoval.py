import argparse
import copy
import hashlib
import itertools
import json
import os
import random
import shutil
import time

import numpy as np
import torch

from miscellaneous.domainbed import command_launchers

import tqdm
import shlex

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess


def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    try:
        # Get list of GPUs from env, split by ',' and remove empty string ''
        # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        # If the env variable is not set, we use all GPUs
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
    n_gpus = len(available_gpus)
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher
}

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, input_dir, infer_dir, output_dir, pool_file):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['input_dir'] = input_dir
        self.train_args['infer_dir'] = infer_dir + '/' + str(self.train_args['batch_size'])
        self.train_args['output_dir'] = self.output_dir
        self.train_args['pretrain_model_list'] = pool_file

        # LODO version: spec_lodoval.py
        command = ['CUDA_VISIBLE_DEVICES=0', 'python', 'spec_lodoval.py']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, bool):
                if v:
                    command.append(f'--{k}')
                    continue
            elif isinstance(v, list):
                # e.g. domainbed_test_envs [0, 1] -> "--domainbed_test_envs 0 1"
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['domainbed_dataset'],
            self.train_args['domainbed_test_envs'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def make_args_list(n_trials, dataset_names, n_hparams_from, n_hparams, n_envs=4, **action_store_args):
    args_list = []
    # LODO: every unordered pair of held-out envs (4 choose 2 = 6 pairs)
    all_held_out_pairs = list(itertools.combinations(range(n_envs), 2))

    for trial_seed in range(n_trials):
        for dataset in dataset_names:
            for test_envs in all_held_out_pairs:
                test_envs = list(test_envs)
                for hparams_seed in range(n_hparams_from, n_hparams):
                    random.seed(hparams_seed)
                    train_args = {}
                    train_args['domainbed_dataset'] = dataset
                    train_args['domainbed_test_envs'] = test_envs
                    train_args['ensemble_weight'] = random.uniform(0.01, 1.0)
                    train_args['child_weight'] = random.uniform(0.01, 1.0)
                    train_args['confidence_weight'] = random.uniform(0.01, 1.0)
                    train_args['batch_size'] = random.choice([32,64,128])
                    train_args['ensemble_lr'] = 10 ** random.uniform(np.log10(1e-4), np.log10(0.1))
                    train_args['clf_lr'] = 1
                    train_args['weight_decay'] = random.choice([1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
                    train_args['hparam_seed'] = hparams_seed
                    train_args['trial_seed'] = trial_seed
                    train_args['seed'] = seed_hash(dataset,
                            tuple(test_envs), hparams_seed, trial_seed)

                    for key, value in action_store_args.items():
                        if value:
                            train_args[str(key)] = value

                    args_list.append(train_args)

    return args_list

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep (LODO validation)')
    parser.add_argument('command', choices=['launch', 'delete_incomplete'])
    parser.add_argument('--domainbed_dataset', nargs='+', type=str, default=['PACS'])
    parser.add_argument('--pretrain_model_list', type=str, default='modelpool_list/sample_list.txt')
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--input_dir', type=str, default='./miscellaneous/domainbed/data/')
    parser.add_argument('--infer_dir', type=str, default="./inferences/")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--mixed_training', action='store_true')
    parser.add_argument('--command_launcher', type=str, required=True)

    parser.add_argument('--skip_confirmation', action='store_true')
    args = parser.parse_args()

    args_list = make_args_list(
        n_trials=args.n_trials,
        dataset_names=args.domainbed_dataset,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        n_envs=args.n_envs,
        mixed_training=args.mixed_training
    )

    jobs = [Job(train_args, args.input_dir, args.infer_dir, args.output_dir, args.pretrain_model_list) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)