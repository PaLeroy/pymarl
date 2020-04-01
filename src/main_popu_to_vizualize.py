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
from matchmaking import REGISTRY as m_REGISTRY

from components.episode_buffer import ReplayBuffer, ReplayBufferPopulation
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
    run_population_test(args=args, logger=logger)

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

def run_population_test(args, logger):
    # Creation of the agents dictionary
    agent_dict = {}
    agent_id = 0
    for i in range(args.n_agent_type):
        args_this_agent = args.__dict__["agent_type_" + str(i + 1)]
        n_agent_this_type = args_this_agent['number']

        assert n_agent_this_type == len(args_this_agent["save_model"])
        save_model = args_this_agent["save_model"]

        assert n_agent_this_type == len(args_this_agent["save_model_interval"])
        save_model_interval = args_this_agent["save_model_interval"]

        assert n_agent_this_type == len(args_this_agent["checkpoint_path"])
        checkpoint_path = args_this_agent["checkpoint_path"]

        assert n_agent_this_type == len(args_this_agent["load_step"])
        load_step = args_this_agent["load_step"]

        for j in range(n_agent_this_type):
            args_this_agent_modified = args_this_agent.copy()
            args_this_agent_modified["save_model"] = save_model[j]
            args_this_agent_modified["save_model_interval"] = \
                save_model_interval[j]
            args_this_agent_modified["checkpoint_path"] = checkpoint_path[j]
            args_this_agent_modified["load_step"] = load_step[j]
            new_agent = {
                'id': agent_id,
                'args_sn': SN(**args_this_agent_modified)
            }
            agent_dict[agent_id] = new_agent
            agent_id += 1

    runner = r_REGISTRY[args.runner](args=args, logger=logger,
                                     agent_dict=agent_dict)

    env_info = runner.get_env_info()
    print("env_info", env_info)
    # Take care that env info is made for 2 teams (-> obs is a tuple)

    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    scheme_buffer = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"][0], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],),
                          "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }

    groups = {
        "agents": args.n_agents,
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBufferPopulation(scheme_buffer,
                                    groups,
                                    args.buffer_size,
                                    env_info["episode_limit"] + 1,
                                    agent_dict,
                                    preprocess=preprocess,
                                    device="cpu"
                                    if args.buffer_cpu_only else args.device)
    match_maker = m_REGISTRY[args.matchmaking](agent_dict)
    print(scheme_buffer)
    for k, v in agent_dict.items():
        agent_dict[k]['args_sn'].n_agents = env_info["n_agents"]
        agent_dict[k]['args_sn'].n_actions = env_info["n_actions"]
        agent_dict[k]['args_sn'].state_shape = env_info["state_shape"]
        agent_dict[k]['args_sn'].use_cuda = args.use_cuda
        agent_dict[k]['mac'] = mac_REGISTRY[agent_dict[k]['args_sn'].mac](
            buffer.scheme,
            groups,
            agent_dict[k]['args_sn'])
        agent_dict[k]['learner'] \
            = le_REGISTRY[agent_dict[k]['args_sn'].learner](
            agent_dict[k]['mac'],
            buffer.scheme,
            logger, agent_dict[k]['args_sn'], id_agent=str(k))
        agent_dict[k]['t_total'] = 0
        agent_dict[k]['model_save_time'] = 0

        if args.use_cuda:
            agent_dict[k]['learner'].cuda()

        checkpoint_path_ = agent_dict[k]['args_sn'].checkpoint_path
        agent_dict[k]["load_timesteps"] = []
        if checkpoint_path_ != "":

            if not os.path.isdir(checkpoint_path_):
                logger.console_logger.info(
                    "Checkpoint directory {} doesn't exist".format(
                        checkpoint_path_))
                return

            # Go through all files in args.checkpoint_path
            for name in os.listdir(checkpoint_path_):
                full_name = os.path.join(checkpoint_path_, name)
                # Check if they are dirs the names of which are numbers

                if os.path.isdir(full_name) and name.isdigit():
                    agent_dict[k]["load_timesteps"].append(int(name))

        else:
            logger.console_logger.info("Checkpoint directory doesn't exist")
            exit()
    if args.matchmaking == "single":
        agent_dict[0]["load_timesteps"]=sorted(agent_dict[0]["load_timesteps"])
        for idx_, timestep_to_load in enumerate(agent_dict[0]["load_timesteps"]):
            if timestep_to_load < 3000000:
                continue
            print("timestep_to_load", timestep_to_load)
            model_path = os.path.join(agent_dict[0]['args_sn'].checkpoint_path,
                                      str(timestep_to_load))

            logger.console_logger.info(
                "Loading model from {}".format(model_path))
            agent_dict[0]['learner'].load_models(model_path)
            agent_dict[0]['t_total'] = timestep_to_load
            if args.evaluate or args.save_replay:
                evaluate_sequential(args, runner)
                return

            runner.setup(scheme=scheme_buffer, groups=groups,
                         preprocess=preprocess)

            for _ in range(args.n_epsiode_per_test):
                # Run for a whole episode at a time
                list_episode_matches = match_maker.list_combat(agent_dict,
                                                               n_matches=args.batch_size_run)
                runner.setup_agents(list_episode_matches, agent_dict)
                episode_batches, total_times, win_list = runner.run(
                    test_mode=True)
                print(win_list)

    if args.matchmaking == "duo":
        agent_dict[0]["load_timesteps"] = sorted(
            agent_dict[0]["load_timesteps"])
        for idx_, timestep_to_load in enumerate(
                agent_dict[0]["load_timesteps"]):
            # if timestep_to_load < 4000000:
            #     continue
            print("timestep_to_load", timestep_to_load)
            model_path1 = os.path.join(agent_dict[0]['args_sn'].checkpoint_path,
                                      str(timestep_to_load))
            logger.console_logger.info("Loading model from {}".format(model_path1))
            agent_dict[0]['learner'].load_models(model_path1)
            agent_dict[0]['t_total'] = timestep_to_load

            model_path2 = os.path.join(agent_dict[1]['args_sn'].checkpoint_path,
                str(timestep_to_load))
            logger.console_logger.info(
                "Loading model from {}".format(model_path2))
            agent_dict[1]['learner'].load_models(model_path2)
            agent_dict[1]['t_total'] = timestep_to_load

            if args.evaluate or args.save_replay:
                evaluate_sequential(args, runner)
                return

            runner.setup(scheme=scheme_buffer, groups=groups,
                         preprocess=preprocess)

            for _ in range(args.n_epsiode_per_test):
                # Run for a whole episode at a time
                list_episode_matches = match_maker.list_combat(agent_dict,
                                                               n_matches=args.batch_size_run)
                runner.setup_agents(list_episode_matches, agent_dict)
                episode_batches, total_times, win_list = runner.run(
                    test_mode=True)
                print(win_list)
    else:
        logger.console_logger.info("Unknown matchmaking")
        exit()

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