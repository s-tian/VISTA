"""
The main script for evaluating and comparing two policies in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    compare_agent (str): path to second saved checkpoint pth file for comparison

    horizon (int): if provided, override maximum horizon of rollout from the one
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate and compare two policies with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.

    python compare_robomimic_trained_policies.py --agent /path/to/model1.pth \
        --compare_agent /path/to/model2.pth --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 --camera_names agentview robot0_eye_in_hand

    # Write the 50 agent rollouts to a new dataset hdf5.

    python compare_robomimic_trained_policies.py --agent /path/to/model1.pth \
        --compare_agent /path/to/model2.pth --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python compare_robomimic_trained_policies.py --agent /path/to/model1.pth \
        --compare_agent /path/to/model2.pth --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
import tqdm
import os
import matplotlib.pyplot as plt

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from robomimic.envs.wrappers import RandomizedCameraWrapper
from vipl.utils.env_utils import get_camera_info


def rollout(policy1, policy2, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, use_existing_state_dict=None):
    """
    Helper function to carry out rollouts and compare two policies. Supports on-screen rendering, off-screen rendering to a video,
    and returns the rollout trajectory and action comparison.

    Args:
        policy1 (instance of RolloutPolicy): first policy loaded from a checkpoint
        policy2 (instance of RolloutPolicy): second policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectory.
            They are excluded by default because the low-dimensional simulation states should be a minimal
            representation of the environment.
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
        action_errors (list): list of L2 errors between actions from the two policies
    """
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy1, RolloutPolicy)
    assert isinstance(policy2, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy1.start_episode()
    policy2.start_episode()
    obs = env.reset()
    if use_existing_state_dict:
        state_dict = use_existing_state_dict
    else:
        state_dict = env.get_state()
    init_state_dict = deepcopy(state_dict)
    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    action_errors = []
    cosines = []
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):

            # get actions from both policies
            act1 = policy1(ob=obs)
            act2 = policy2(ob=obs)

            # compute L2 error between actions
            l2_error = np.linalg.norm(act1 - act2)
            # Normalize actions to compute cosine similarity
            norm_act1 = act1 / np.linalg.norm(act1) if np.linalg.norm(act1) != 0 else act1
            norm_act2 = act2 / np.linalg.norm(act2) if np.linalg.norm(act2) != 0 else act2
            cosine_similarity = np.dot(norm_act1, norm_act2)
            cosines.append(cosine_similarity)
            
            action_errors.append(l2_error)
            # if l2_error > 1.5:
            #     print("Action from policy 1: ", act1)
            #     print("Action from policy 2: ", act2)
            #     print("Err: ", l2_error)
            # play action from the first policy
            next_obs, r, done, _ = env.step(act1)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act1)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))
    traj["success"] = float(success)

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj, action_errors, cosines, init_state_dict


def run_trained_agent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video)  # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative paths to agents
    ckpt_path1 = args.agent
    ckpt_path2 = args.compare_agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policies
    policy1, ckpt_dict1 = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path1, device=device, verbose=True)
    policy2, ckpt_dict2 = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path2, device=device, verbose=True)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict1)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict1,
        env_name=args.env,
        render=args.render,
        render_offscreen=(args.video_path is not None),
        verbose=True,
    )

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # if randomizing camera, wrap environment
    if args.randomize_camera:
        env = RandomizedCameraWrapper(
            env,
            camera_name=args.camera_name,
            camera_pos_std=args.camera_pos_std,
            camera_angle_std=args.camera_angle_std,
            randomize_freq=args.samples_per_view*2,
        )

    # maybe create video writer
    video_writer1 = None
    video_writer2 = None
    if write_video:
        video_writer1 = imageio.get_writer(args.video_path.replace('.mp4', '_policy1.mp4'), fps=20)
        video_writer2 = imageio.get_writer(args.video_path.replace('.mp4', '_policy2.mp4'), fps=20)

    # maybe open hdf5 to write rollouts for the first policy only
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        # create required dirs
        os.makedirs(os.path.dirname(args.dataset_path), exist_ok=True)
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    all_action_errors = []
    all_cosines = []
    for i in tqdm.tqdm(range(rollout_num_episodes)):
        print("Policy 1 is trained on simulated data:")
        stats1, traj, action_errors1, cosines1, initial_state_dict = rollout(
            policy1=policy1,
            policy2=policy2,
            env=env,
            horizon=rollout_horizon,
            render=args.render,
            video_writer=video_writer1,
            video_skip=args.video_skip,
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
        )
        print("Policy 2 is trained on simulated data:")
        stats2, traj2, action_errors2, cosines2, _ = rollout(
            policy1=policy2,
            policy2=policy1,
            env=env,
            horizon=rollout_horizon,
            render=args.render,
            video_writer=video_writer2,
            video_skip=args.video_skip,
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
            use_existing_state_dict=initial_state_dict
        )
        rollout_stats.append(stats1)
        all_action_errors.append(action_errors1)
        all_cosines.append(cosines1)

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]  # model xml for this episode
            camera_info = get_camera_info(
                env.env,
                camera_names=args.camera_names,
                camera_height=512,
                camera_width=512
            )
            ep_data_grp.attrs["success"] = traj["success"]  # success flag for this episode
            ep_data_grp.attrs["camera_info"] = json.dumps(camera_info, indent=4)  # camera info for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]  # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = {k: np.mean(rollout_stats[k]) for k in rollout_stats}
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer1.close()
        video_writer2.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)  # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))

    # Plot L2 error between actions from the two policies over time
    plt.figure()
    # Calculate mean and standard deviation for error bars
    
    for idx, (errors, cosines) in enumerate(zip(all_action_errors, all_cosines)):
        plt.figure()
        time_steps = np.arange(len(errors))
        plt.plot(time_steps, errors, '-o', label=f'Error Series {idx + 1}')
        plt.plot(time_steps, cosines, '-x', label=f'Cosine Similarity Series {idx + 1}')
        plt.xlabel('Time Step')
        plt.ylabel('L2 Error between Actions')
        plt.title(f'L2 Error between Actions from Two Policies Over Time (Series {idx + 1})')
        plt.legend()
        plt.savefig(f'05_20_expl/L2_Error_Series_{idx + 1}.png')
        plt.close()

    env.env.sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    parser.add_argument(
        "--compare_agent",
        type=str,
        required=True,
        help="path to second saved checkpoint pth file for comparison",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=27,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    parser.add_argument(
        "--randomize_camera",
        help="randomize camera position",
        action='store_true'
    )

    parser.add_argument(
        "--camera_name",
        type=str,
        help="camera name",
        default="agentview"
    )

    parser.add_argument(
        "--camera_angle_std",
        type=float,
        help="camera angle std",
        default=0.075
    )

    parser.add_argument(
        "--camera_pos_std",
        type=float,
        help="camera position std",
        default=0.03
    )

    parser.add_argument(
        "--samples_per_view",
        type=int,
        help="number of rollouts to run on each randomized view",
        default=1
    )

    args = parser.parse_args()
    run_trained_agent(args)

