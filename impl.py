import copy
import json
import os
import time
from collections import namedtuple
from typing import List
from datetime import datetime
import dowel
import gym
import higher
import hydra
import argparse
import metaworld
import torch
import torch.distributions as D
import torch.optim as O
from dowel import tabular, logger
from garage import log_multitask_performance, EpisodeBatch
from garage.envs import MetaWorldSetTaskEnv
from garage.experiment import MetaWorldTaskSampler, SetTaskSampler
from garage.experiment.deterministic import set_seed, get_seed
from garage.sampler import WorkerFactory, LocalSampler, RaySampler
from hydra.utils import get_original_cwd
from helpers import environmentvariables

from losses import policy_loss_on_batch, vf_loss_on_batch
from nn import MLP
from utils import Experience
from utils import ReplayBuffer
from worker import CustomWorker

environmentvariables.initialize()


def rollout_policy(policy: MLP, env, render: bool = False) -> List[Experience]:
    trajectory = []
    state, _ = env.reset()
    if render:
        env.render(mode='human')
    done = False
    total_reward = 0
    episode_t = 0
    success = False
    policy.eval()
    current_device = list(policy.parameters())[-1].device
    while not done:
        with torch.no_grad():
            action_sigma = 0.2
            action = policy(torch.from_numpy(state).to(current_device).float()).squeeze()

            action_dist = D.Normal(action, torch.empty_like(action).fill_(action_sigma))
            log_prob = action_dist.log_prob(action).to("cpu").numpy().sum()

            np_action = action.squeeze().cpu().numpy()
            np_action = np_action.clip(min=env.action_space.low, max=env.action_space.high)

        env_step = env.step(np_action)
        next_state, reward, done, info_dict = env_step.observation, env_step.reward, env_step.terminal or env_step.timeout, env_step.env_info

        if "success" in info_dict and info_dict["success"]:
            success = True

        if render:
            env.render(mode='human')
        trajectory.append(Experience(state, np_action, next_state, reward, done))
        state = next_state
        total_reward += reward
        episode_t += 1
        if episode_t >= env._current_env.max_path_length or done:
            break

    return trajectory, total_reward, success


def build_networks_and_buffers(args, env, task_config):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy_head = [32, 1] if args.advantage_head_coef is not None else None
    policy = MLP(
        [obs_dim] + [args.net_width] * args.net_depth + [action_dim],
        final_activation=torch.tanh,
        extra_head_layers=policy_head,
        w_linear=args.weight_transform,
    ).to(args.device)

    vf = MLP(
        [obs_dim] + [args.net_width] * args.net_depth + [1],
        w_linear=args.weight_transform,
    ).to(args.device)

    train_buffers = read_buffers(action_dim, args, obs_dim, task_config.train_tasks,
                                 os.path.join(os.getenv("BUFFER_DIR"), "{}.hdf5"))
    test_buffers = read_buffers(action_dim, args, obs_dim, task_config.test_tasks,
                                os.path.join(os.getenv("BUFFER_DIR"), "{}.hdf5"))

    return policy, vf, train_buffers, test_buffers


def read_buffers(action_dim, args, obs_dim, task_numbers, buffer_paths):
    # Load task to index mapping
    path_env_mapping = "../../../config/env_mapping_sac_training.json"
    with open(path_env_mapping, 'r') as mapping_data_file:
        env_name_to_idx = json.load(mapping_data_file)

    idx_to_env_name = {v: k for k, v in env_name_to_idx.items()}

    buffer_paths = [
        buffer_paths.format(idx) for idx in task_numbers
    ]

    buffers = dict()

    for i, task_idx in enumerate(task_numbers):
        buffers[idx_to_env_name[task_idx]] = \
            ReplayBuffer(
                args.inner_buffer_size,
                obs_dim,
                action_dim,
                discount_factor=args.discount,
                immutable=True,
                load_from=buffer_paths[i],
            )

    return buffers


def get_opts_and_lrs(args, policy, vf):
    policy_opt = O.Adam(policy.parameters(), lr=args.outer_policy_lr)
    vf_opt = O.Adam(vf.parameters(), lr=args.outer_value_lr)
    policy_lrs = [
        torch.nn.Parameter(torch.tensor(args.inner_policy_lr).to(args.device))
        for p in policy.parameters()
    ]
    vf_lrs = [
        torch.nn.Parameter(torch.tensor(args.inner_value_lr).to(args.device))
        for p in vf.parameters()
    ]

    return policy_opt, vf_opt, policy_lrs, vf_lrs


def get_metaworld_env(env_name: str = 'ml10', env_ml1_name: str = ''):
    if env_name == 'ml1':
        ml = metaworld.ML1(env_name=env_ml1_name)
    elif env_name == 'ml10':
        ml = metaworld.ML10()
    elif env_name == 'ml45':
        ml = metaworld.ML45()
    else:
        raise NotImplementedError()

    train_sampler = MetaWorldTaskSampler(ml, 'train')
    env_specs = train_sampler.sample(len(ml.train_classes))[0]()  # sample one task instance to get environment specs

    test_sampler = SetTaskSampler(
        MetaWorldSetTaskEnv,
        env=MetaWorldSetTaskEnv(ml, 'test'),
    )

    return env_specs, train_sampler, test_sampler


def soft_update(source, target, args):
    for param_source, param_target in zip(source.named_parameters(), target.named_parameters()):
        assert param_source[0] == param_target[0]
        param_target[1].data = args.target_vf_alpha * param_target[1].data + (1 - args.target_vf_alpha) * param_source[1].data


@hydra.main(config_path="config", config_name="config.yaml")
def run(args):
    with open(f"{get_original_cwd()}/{args_v2.task_config}", "r") as f:
        task_config = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )

    log_dir = os.path.join(os.getenv("OUT_DIR"), datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + "_MACAW")
    os.makedirs(log_dir)
    setup_logger(log_dir)

    logger.log(f"Logging to {log_dir}")

    start_time = time.time()

    total_train_env_steps = 0

    if args.advantage_head_coef == 0:
        args.advantage_head_coef = None

    # env = MetaworldEnv(include_goal=args.include_goal)

    seed = args.seed if args.seed is not None else 1

    set_seed(seed)  # set seed directly before we get envs
    env_specs, train_task_sampler, test_task_sampler = get_metaworld_env(env_name=task_config.env,
                                                                         env_ml1_name=task_config.env_ml1_name if hasattr(task_config,
                                                                                                                          'env_ml1_name') else '')

    policy, vf, task_buffers, test_buffers = build_networks_and_buffers(args, env_specs, task_config)
    policy_opt, vf_opt, policy_lrs, vf_lrs = get_opts_and_lrs(args, policy, vf)

    start_itr = 1

    # load a snapshot
    if args.snapshot_load:
        start_itr, policy_state, policy_opt_state, vf_state, vf_opt_state = Snapshotter.load_snapshot(path=args.snapshot_path)
        start_itr += 1  # start with the next iteration
        policy.load_state_dict(policy_state)
        policy_opt.load_state_dict(policy_opt_state)
        vf.load_state_dict(vf_state)
        vf_opt.load_state_dict(vf_opt_state)

    logger.log("Start training ...")

    for train_step_idx in range(start_itr, int(5.1e6)):
        itr_start_time = time.time()
        for i, (train_task_idx, task_buffer) in enumerate(
                zip(task_config.train_tasks, task_buffers.values())
        ):
            inner_batch = task_buffer.sample(
                args.inner_batch_size, return_dict=True, device=args.device
            )
            outer_batch = task_buffer.sample(
                args.outer_batch_size, return_dict=True, device=args.device
            )

            # Adapt value function
            opt = O.SGD([{"params": p, "lr": None} for p in vf.parameters()])
            with higher.innerloop_ctx(
                    vf, opt, override={"lr": vf_lrs}, copy_initial_weights=False
            ) as (f_vf, diff_value_opt):
                loss = vf_loss_on_batch(f_vf, inner_batch, inner=True)
                diff_value_opt.step(loss)

                meta_vf_loss = vf_loss_on_batch(f_vf, outer_batch)
                total_vf_loss = meta_vf_loss / len(task_config.train_tasks)
                total_vf_loss.backward()

            # Adapt policy using adapted value function
            adapted_vf = f_vf
            opt = O.SGD([{"params": p, "lr": None} for p in policy.parameters()])
            with higher.innerloop_ctx(
                    policy, opt, override={"lr": policy_lrs}, copy_initial_weights=False
            ) as (f_policy, diff_policy_opt):
                loss = policy_loss_on_batch(
                    f_policy,
                    adapted_vf,
                    inner_batch,
                    args.advantage_head_coef,
                    inner=True,
                )

                diff_policy_opt.step(loss)

                meta_policy_loss = policy_loss_on_batch(
                    f_policy, adapted_vf, outer_batch, args.advantage_head_coef
                )

                (meta_policy_loss / len(task_config.train_tasks)).backward()

                # Sample adapted policy trajectory, add to replay buffer i [L12]
                # if train_step_idx % args.rollout_interval == 0:
                #     adapted_trajectory, adapted_reward, success = rollout_policy(
                #         f_policy, env
                #     )
                #     print("train_step", train_step_idx, " rewards", adapted_reward, " success", success)

            total_train_env_steps += args.inner_batch_size + args.outer_batch_size

        if train_step_idx % args.epoch_interval == 0:
            # evaluation on test set
            logger.log("Start eval ...")
            n_exploration_eps = 10

            eval_model(args=args,
                       n_exploration_eps=n_exploration_eps,
                       policy=policy,
                       policy_lrs=policy_lrs,
                       test_buffers=test_buffers,
                       test_task_sampler=test_task_sampler,
                       train_step_idx=train_step_idx,
                       vf=vf,
                       vf_lrs=vf_lrs,
                       n_test_tasks=None)

            # log stats
            logger.log('Time %.2f s' % (time.time() - start_time))
            logger.log('EpochTime %.2f s' % (time.time() - itr_start_time))
            tabular.record('TotalEnvSteps', total_train_env_steps)
            tabular.record('TrainSteps', train_step_idx)
            logger.log(tabular)

            logger.dump_all(train_step_idx)
            tabular.clear()

            # save checkpoint
            Snapshotter.save_snapshot(log_dir, train_step_idx, policy, policy_opt, vf, vf_opt)


def setup_logger(log_dir):
    tabular_log_file = os.path.join(log_dir, 'progress.csv')
    text_log_file = os.path.join(log_dir, 'debug.log')
    logger.add_output(dowel.TextOutput(text_log_file))
    logger.add_output(dowel.CsvOutput(tabular_log_file))
    logger.add_output(dowel.TensorBoardOutput(log_dir, x_axis='TotalEnvSteps'))
    logger.add_output(dowel.StdOutput())


def eval_model(args, n_exploration_eps, policy, policy_lrs, test_buffers, test_task_sampler, train_step_idx, vf, vf_lrs, n_test_tasks=None):
    """
    Based on garage MetaEvaluator

    """
    eval_policy = copy.deepcopy(policy)

    nr_test_tasks = test_task_sampler.n_tasks if n_test_tasks is None else n_test_tasks

    env_instances = test_task_sampler.sample(nr_test_tasks)
    env = env_instances[0]()

    max_episode_length = env.spec.max_episode_length

    worker_factory = WorkerFactory(seed=get_seed(),
                                   max_episode_length=max_episode_length,
                                   n_workers=n_exploration_eps,
                                   worker_class=CustomWorker,
                                   worker_args={})

    if args.device == 'cpu':
        test_episode_sampler = RaySampler.from_worker_factory(worker_factory=worker_factory,
                                                              agents=eval_policy,
                                                              envs=env)
    else:
        # choose the local sampler on the gpu
        test_episode_sampler = LocalSampler.from_worker_factory(worker_factory=worker_factory,
                                                                agents=eval_policy,
                                                                envs=env)

    adapted_episodes = list()

    for env_instance in env_instances:
        # reset policy and value function
        eval_policy = copy.deepcopy(policy)
        eval_value_function = copy.deepcopy(vf)

        # offline update of value function and policy
        env_name = env_instance._task['inner'].env_name

        # map env_name to test buffer index
        test_buffer = test_buffers[env_name]

        value_batch_dict = test_buffer.sample(args.eval_batch_size, return_dict=True, device=args.device)
        policy_batch_dict = value_batch_dict

        opt = O.SGD([{'params': p, 'lr': None} for p in eval_value_function.adaptation_parameters()])
        with higher.innerloop_ctx(eval_value_function, opt, override={'lr': vf_lrs}) as (f_value_function, diff_value_opt):
            loss = vf_loss_on_batch(f_value_function, value_batch_dict, inner=True)
            diff_value_opt.step(loss)

            # Soft update target value function parameters
            # self.soft_update(f_value_function, vf_target) TODO use this soft update?

            policy_opt = O.SGD([{'params': p, 'lr': None} for p in eval_policy.adaptation_parameters()])
            with higher.innerloop_ctx(eval_policy, policy_opt, override={'lr': policy_lrs}) as (f_policy, diff_policy_opt):
                loss = policy_loss_on_batch(f_policy, f_value_function, policy_batch_dict,
                                            adv_coef=args.advantage_head_coef,
                                            inner=True)
                diff_policy_opt.step(loss)

                # extract updated policy
                adapted_policy = eval_policy
                adapted_policy.load_state_dict(f_policy.state_dict())
                for variable in adapted_policy.parameters():
                    variable.detach_()

        # obtain episodes on the current task instance
        with torch.no_grad():
            adapted_policy.eval()
            adapted_eps = test_episode_sampler.obtain_samples(0,
                                                              num_samples=max_episode_length * n_exploration_eps,
                                                              agent_update=adapted_policy,
                                                              env_update=env_instance)
        # add adapted episodes
        adapted_episodes.append(adapted_eps)

    # log evaluation stats
    with tabular.prefix('MetaTest' + '/'):
        log_multitask_performance(
            train_step_idx,
            EpisodeBatch.concatenate(*adapted_episodes),
            discount=args.discount,
            name_map=None)


class Snapshotter:

    @staticmethod
    def load_snapshot(path):
        if not os.path.isabs(path):
            path = os.path.join(get_original_cwd(), path)

        files = [file for file in os.listdir(path)
                 if os.path.isfile(os.path.join(path, file)) and file.startswith('checkpoint') and file.endswith('.pth')]

        # sort to get the latest checkpoint
        files.sort()
        checkpoint_file = files[-1]

        checkpoint_file_path = os.path.join(path, checkpoint_file)

        logger.log(f'Loading snapshot {checkpoint_file_path}')

        snapshot = torch.load(checkpoint_file_path)

        return snapshot['train_step_idx'], \
               snapshot['policy_state_dict'], \
               snapshot['policy_optimizer_state_dict'], \
               snapshot['value_function_state_dict'], \
               snapshot['value_function_optimizer_state_dict'],

    @staticmethod
    def save_snapshot(log_dir, train_step_idx, policy, policy_opt, vf, vf_opt):
        file_name = Snapshotter.get_file_name(train_step_idx)

        logger.log(f"Saving snapshot {file_name}")

        torch.save({
            'train_step_idx': train_step_idx,
            'policy_state_dict': policy.state_dict(),
            'policy_optimizer_state_dict': policy_opt.state_dict(),
            'value_function_state_dict': vf.state_dict(),
            'value_function_optimizer_state_dict': vf_opt.state_dict(),
        }, os.path.join(log_dir, file_name))

    @staticmethod
    def get_file_name(train_step_idx):
        return f'checkpoint_{train_step_idx:06d}.pth'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argumentparser")
    parser.add_argument('--task_config', default='task_config/metaworld_ml10.json',
                        type=str, help="Path to task config file")
    global args_v2
    args_v2 = parser.parse_args()

    run()
