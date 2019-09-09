import numpy as np
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
from utils.logging import get_logger
import yaml

import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from main import _get_config, recursive_dict_update
from run import args_sanity_check, evaluate_sequential

SETTINGS[
    'CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log, env_args):
    # Setting the random seed throughout the modules
    np.random.seed(_config["seed"])
    th.manual_seed(_config["seed"])
    env_args['seed'] = _config["seed"]

    # run the framework
    run_test(_run, _config, _log)


def run_test(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))),
                                     "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential_test(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def run_sequential_test(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()

    if args.multi:
        args.n_agents_team1 = env_info["n_agents"]
        args.n_agents_team2 = env_info["n_enemies"]
        args.n_agents = env_info["n_agents"] + env_info["n_enemies"]
    else:
        args.n_agents = env_info["n_agents"]

    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    if args.multi:
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs_team_1": {"vshape": env_info["obs_shape"][0],
                           "group": "agents_team_1"},
            "obs_team_2": {"vshape": env_info["obs_shape"][1],
                           "group": "agents_team_2"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],),
                              "group": "agents", "dtype": th.int},
            "reward": {"vshape": (args.n_agents,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }

        groups = {
            "agents": args.n_agents,
            "agents_team_1": args.n_agents_team1,
            "agents_team_2": args.n_agents_team2
        }
    else:
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],),
                              "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,) if not args.multi else (args.n_agents,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }

        groups = {
            "agents": args.n_agents
        }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    buffer = ReplayBuffer(scheme, groups, args.buffer_size,
                          env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directory {} doesn't exist".format(
                    args.checkpoint_path))
            return
        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))
        timesteps = sorted(timesteps)
        for idx_, timestep_to_load in enumerate(timesteps):
            if idx_ % args.n_skip != 0:
                continue
            model_path = os.path.join(args.checkpoint_path,
                                      str(timestep_to_load))

            logger.console_logger.info(
                "Loading model from {}".format(model_path))
            learner.load_models(model_path)
            runner.t_env = timestep_to_load

            if args.evaluate or args.save_replay:
                evaluate_sequential(args, runner)
                return

            for _ in range(args.n_epsiode_per_test):
                # Run for a whole episode at a time
                episode_batch = runner.run(test_mode=True)

        runner.close_env()
        logger.console_logger.info("Finished testing")

    else:
        logger.console_logger.info("Checkpoint directory doesn't exist")


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(
            os.path.join(os.path.dirname(__file__), "config", "default.yaml"),
            "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
