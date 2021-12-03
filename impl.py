import copy
import json
import os
import time
from collections import namedtuple
from itertools import count
from typing import List

import dowel
import gym
import higher
import hydra
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
from tqdm import tqdm

from losses import policy_loss_on_batch, vf_loss_on_batch
from nn import MLP
from utils import Experience
from utils import ReplayBuffer
from worker import CustomWorker

# TODO remove!
# temporary: do not print warning
gym.logger.set_level(40)


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

    train_buffers = read_buffers(action_dim, args, obs_dim, task_config.train_tasks, task_config.train_buffer_paths)
    test_buffers = read_buffers(action_dim, args, obs_dim, task_config.test_tasks, task_config.test_buffer_paths)

    return policy, vf, train_buffers, test_buffers


def read_buffers(action_dim, args, obs_dim, task_numbers, buffer_paths):
    buffer_paths = [
        buffer_paths.format(idx) for idx in task_numbers
    ]

    buffers = [
        ReplayBuffer(
            args.inner_buffer_size,
            obs_dim,
            action_dim,
            discount_factor=args.discount,
            immutable=True,
            load_from=buffer_paths[i],
        )
        for i, task in enumerate(task_numbers)
    ]

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
    env_specs = train_sampler.sample(1)[0]()  # sample one task instance to get environment specs

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
    with open(f"{get_original_cwd()}/{args.task_config}", "r") as f:
        task_config = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )

    log_dir = './out'  # TODO set to log dir
    os.makedirs(log_dir)
    setup_logger(log_dir)

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

    for train_step_idx in count(start=1):
        itr_start_time = time.time()
        for i, (train_task_idx, task_buffer) in enumerate(
                zip(task_config.train_tasks, task_buffers)
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

        target_train_steps_reached = False
        if total_train_env_steps >= 6e6:  # reached target environment steps
            target_train_steps_reached = True

        if train_step_idx % args.epoch_interval == 0 or target_train_steps_reached:
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
            logger.log(tabular)

            logger.dump_all(train_step_idx)
            tabular.clear()

            # save checkpoint
            checkpoint_file_name = 'checkpoint_' + str(train_step_idx) + '.pth'

            torch.save({
                'train_step_idx': train_step_idx,
                'policy_state_dict': policy.state_dict(),
                'policy_optimizer_state_dict': policy_opt.state_dict(),
                'value_function_state_dict': vf.state_dict(),
                'value_function_optimizer_state_dict': vf_opt.state_dict(),
            }, os.path.join(log_dir, checkpoint_file_name))

        if target_train_steps_reached:
            break


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

    looper = tqdm(env_instances)
    for env_instance in looper:
        # reset policy and value function
        eval_policy = copy.deepcopy(policy)
        eval_value_function = copy.deepcopy(vf)

        # offline update of value function and policy
        env_name = env_instance._task['inner'].env_name

        # TODO map env_name to test buffer index
        env_idx = 0
        test_buffer = test_buffers[env_idx]

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

                # obtain episodes on the current task instance

                with torch.no_grad():
                    # TODO is this really working and using the updated parameters of f_policy?
                    adapted_policy = eval_policy
                    adapted_policy.load_state_dict(f_policy.state_dict())
                    for variable in adapted_policy.parameters():
                        variable.detach_()

                    adapted_policy.eval()
                    adapted_eps = test_episode_sampler.obtain_samples(0,
                                                                      num_samples=max_episode_length * n_exploration_eps,
                                                                      agent_update=adapted_policy,
                                                                      env_update=env_instance)
                    # env = env_instance()
                    # eps_rewards = list()
                    # eps_success = list()
                    # for eps in range(n_exploration_eps):
                    #     adapted_trajectory, adapted_reward, success = rollout_policy(f_policy, env, render=args.render)
                    #     eps_rewards.append(adapted_reward)
                    #     eps_success.append(success)

                # add adapted episodes
                adapted_episodes.append(adapted_eps)

    # log evaluation stats
    with tabular.prefix('MetaTest' + '/'):
        log_multitask_performance(
            train_step_idx,
            EpisodeBatch.concatenate(*adapted_episodes),
            discount=args.discount,
            name_map=None)


if __name__ == "__main__":
    run()
