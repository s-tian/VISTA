import os
import copy
import time
import argparse
import tqdm
from PIL import Image, ImageDraw, ImageFont
import PIL
import h5py

import numpy as np
import torch
from robomimic.envs.env_robosuite import EnvRobosuite
from robosuite.utils.camera_utils import CameraMover, generate_random_camera_pose, generate_random_camera_pose_on_sphere, get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

from vipl.utils.camera_pose_sampler import CameraPoseSampler

try:
    import mimicgen_envs
except Exception as e:
    print("You probably don't have mimicgen environments installed, which you might need :/")

from vipl.utils.cam_utils import posori_to_rotmat
from vipl.utils.exp_utils import model_to_label
from vipl.utils.viz_utils.camera_pose_visualizer import CameraPoseVisualizer
from vipl.models.augmentation import get_model_by_name


def get_demo_index_from_key(key):
    return int(key.split("_")[1])

def from_file_state_gen(dataset):
    f = h5py.File(dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    for ind in range(len(demos)):
        # calculate the episode number based on the current iteration through the dataset and the demo iter
        # output_ind = parse_iter * len(all_demos) + ind
        # output_ep_name = "demo_{}".format(output_ind)
        # output_ep_name = demos[ind]
        # actual index of the demo in the original file, e.g. demo_6 -> 6. 
        # This is not necessarily the same as the "ind" iterator
        demo_index = get_demo_index_from_key(demos[ind])
        output_ep_idx = demo_index
        ep_to_read = demos[ind]    
        states = f["data/{}/states".format(ep_to_read)][()]
        # randomly shuffle states
        np.random.shuffle(states)
        for state in states:
            yield {"states": state}

def infinite_state_gen(state):
    while True:
        yield state

def main(args):
    
    camera_name = "agentview"
    num_trials = 100
    env_name = args.env
    env2 = EnvRobosuite.create_for_data_processing(
        env_name,
        camera_names=["agentview"],
        robots="Panda",
        render=False,
        camera_height=256,
        camera_width=256,
        reward_shaping=False,
        render_offscreen=True,
        use_image_obs=True,
        use_depth_obs=False,
    )
    initial_obs = env2.reset()[f"{camera_name}_image"]
    initial_obs = Image.fromarray(initial_obs.astype(np.uint8))
    # write the initial observation to a file
    initial_obs.save(os.path.join(args.output_dir, "initial_obs_84.png"))
    env = EnvRobosuite.create_for_data_processing(
        env_name,
        camera_names=["agentview"],
        robots="Panda",
        render=False,
        camera_height=256,
        camera_width=256,
        reward_shaping=False,
        render_offscreen=True,
        use_image_obs=True,
        use_depth_obs=True,
    )
    # robot_base_pos = np.copy(env.env.robots[0].base_pos)
    initial_obs_all = env.reset()
    initial_obs = initial_obs_all[f"{camera_name}_image"]
    initial_depth = initial_obs_all[f"{camera_name}_depth"]

    if env_name == "Lift":
        robot_base_pos = env.env.sim.data.body_xpos[env.env.cube_body_id]
    else:
        robot_base_pos = np.array([0.01375589, 0.0, 0.83170968]) # harvested from Lift environment, set y = 0 to center in left/right directions

    print("Robot base pos is ", robot_base_pos)
    initial_env_state = env.get_state()
    del initial_env_state["model"]
    initial_obs = Image.fromarray(initial_obs.astype(np.uint8))
    # write the initial observation to a file
    initial_obs.save(os.path.join(args.output_dir, "initial_obs.png"))

    initial_camera_extrinsics = get_camera_extrinsic_matrix(env.env.sim, camera_name)

    camera_mover = CameraMover(env.env, camera_name)
    default_camera_pose = copy.deepcopy(camera_mover.get_camera_pose())
    # print(default_camera_pose)
    camera_sampler = CameraPoseSampler(sampler_type=args.camera_sampler)
    # set all random seeds
    np.random.seed(args.seed)
    random_camera_poses = camera_sampler.sample_poses(n=num_trials,
                                                      starting_pose=default_camera_pose)
    # print(random_camera_poses)
    # default_camera_position = default_camera_pose[0]

    # radius = np.linalg.norm(default_camera_position - robot_base_pos)

    # random_camera_poses = generate_random_camera_pose_on_sphere(
    #     current_pose=default_camera_pose,
    #     radius=radius, # lookat approximately the cube 
    #     radius_std=0.05,
    #     theta_range=[-np.pi/4, np.pi/4],
    #     # theta_range=[-np.pi/10, np.pi/10],
    #     num_samples=num_trials,
    # )
    if args.input_dataset:
        print(f"Loading states from {args.input_dataset}...")
        state_generator = from_file_state_gen(args.input_dataset) 
    else:
        print("No input dataset provided, loading initial state from the environment")
        state_generator = infinite_state_gen(initial_env_state)

    # for pose in random_camera_poses:
        # pose[0] = default_camera_pose[0]


    # camera_pose_matrices = []
    # for pose in random_camera_poses:
    #     pose_matrix = posori_to_rotmat(pose[0], pose[1])
    #     camera_pose_matrices.append(pose_matrix)
    
    # camera_pos = [c[:, 3] for c in camera_pose_matrices]
    # # compute xlim ylim zlim based on camera views
    # xlim = [np.min([c[0] for c in camera_pos]), np.max([c[0] for c in camera_pos])]
    # ylim = [np.min([c[1] for c in camera_pos]), np.max([c[1] for c in camera_pos])]
    # zlim = [np.min([c[2] for c in camera_pos]), np.max([c[2] for c in camera_pos])]
    # # make the lims 2x wider
    # xlim = [xlim[0] - 0.5 * (xlim[1] - xlim[0]), xlim[1] + 0.5 * (xlim[1] - xlim[0])]
    # ylim = [ylim[0] - 0.5 * (ylim[1] - ylim[0]), ylim[1] + 0.5 * (ylim[1] - ylim[0])]
    # zlim = [zlim[0] - 0.5 * (zlim[1] - zlim[0]), zlim[1] + 0.5 * (zlim[1] - zlim[0])]
    # zlim = [0, 2]
    # visualizer = CameraPoseVisualizer(xlim=xlim, ylim=ylim, zlim=zlim)
    # for v in camera_pose_matrices:
    #     visualizer.extrinsic2pyramid(v, aspect_ratio=1, focal_len_scaled=0.1)
    # visualizer.save("debug.png")

    # random_camera_poses = generate_random_camera_pose(
    #     current_pose=default_camera_pose,
    #     angle_std=0.075,
    #     position_std=0.03,
    #     num_samples=num_trials,
    # )
    # initialize models
    models = {}
    model_inference_times = {}
    for model in args.models:
        kwargs = {}
        if model == "deproj":
            kwargs["depth_range"] = (initial_depth.min(), initial_depth.max())
        models[model] = get_model_by_name(model, **kwargs)

    for model in models.keys():
        model_inference_times[model] = list()

    if args.use_single_state:
        state_to_set = next(state_generator)

    for i in tqdm.tqdm(range(num_trials)):
        # Specify the positions for text labels
        text_positions = [
            (10, 256 + 10),  # Initial image
            (256 + 10 + 10, 256 + 10),  # Simulated
        ]
        text_labels = [
            "Initial image",
            "Simulated",
        ]

        camera_mover.set_camera_pose(
            pos=random_camera_poses[i][0],
            quat=random_camera_poses[i][1],
        )

        perturbed_camera_extrinsics = get_camera_extrinsic_matrix(env.env.sim, camera_name)

        if not args.use_single_state:
            state_to_set = next(state_generator)

        real_env_image = env.reset_to(state=state_to_set)[f"{camera_name}_image"]
        real_env_image = Image.fromarray(real_env_image.astype(np.uint8))

        initial_obs = env2.reset_to(state=state_to_set)[f"{camera_name}_image"]
        initial_obs = Image.fromarray(initial_obs.astype(np.uint8))

        # Create a single image with more columns for each model
        grid = Image.new(
            mode="RGB",
            size=((len(models)+2) * 256 + 20, 256 + 20 + 50),  # Increased the height to accommodate text labels
            color=(255, 255, 255),
        )

        # Draw images on the grid
        draw = ImageDraw.Draw(grid)
        font = ImageFont.load_default(30)
        # save the images into a grid side by side, with the initial observation on the left, then a 10 pixel gap,
        # then the augmented image, then the real environment image
        grid.paste(initial_obs, (0, 0))
        grid.paste(real_env_image, (256 + 10, 0))

        for j, (model_name, model) in enumerate(models.items()):
            start_time = time.time()
            augmented_image = model.augment(
                original_image=initial_obs,
                # cam2worlds in opencv convention
                original_camera = initial_camera_extrinsics,
                target_camera = perturbed_camera_extrinsics,
                convention="opencv",
            )
            end_time = time.time()
            model_inference_times[model_name].append(end_time - start_time)
            augmented_image_filename = os.path.join(args.output_dir, f"{model_name}_trial_{i}.png")
            # if "debug" in model_name:
            #     augmented_image_84 = augmented_image.resize((84, 84))
            # else:
            #     augmented_image_84 = augmented_image.resize((84, 84), resample=Image.LANCZOS)
            augmented_image.save(augmented_image_filename)

            grid.paste(augmented_image, ((j+2) * 256 + 10, 0))
            # then, label the image with text saying "Initial image", "Simulated", and the model name on a row above it
            text_positions.append(((j+2) * 256 + 20 + 10, 256 + 10))  # Model name
            text_labels.append(model_to_label[model_name])
        real_env_image.save(os.path.join(args.output_dir, f"real_env_image_trial_{i}.png"))

        # Calculate and print the average inference time for each model
        for model_name, times in model_inference_times.items():
            average_time = sum(times) / len(times)
            print(f"Average inference time for {model_name}: {average_time:.4f} seconds")

        # Draw text on the grid, make it big
        for position, label in zip(text_positions, text_labels):
            draw.text(position, label, font=font, fill=(0, 0, 0))

        grid.save(os.path.join(args.output_dir, f"trial_{i:03}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add model argument to parser
    parser.add_argument("--models", nargs='+', type=str, default=[])
    parser.add_argument("--env", default="Lift", type=str)
    parser.add_argument("--camera_sampler", default="small_perturb", type=str)
    parser.add_argument("--output_dir", type=str, default="./compare_images_logs", help="Directory to save output files")
    parser.add_argument("--seed", type=int, default=0, help="Random seed to use for selecting camera poses and dataset states, use this to generate views from different models with same input image/poses")
    parser.add_argument("--input_dataset", type=str, default=None, help="Path to the input dataset file")
    # add a boolean flag called use_single_state
    parser.add_argument("--use_single_state", action="store_true", help="Use a single state for all trials")
    args = parser.parse_args()
    main(args)

