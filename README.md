# VISTA 

This repository contains the code for the paper "View-Invariant Policy Learning via Zero-Shot Novel View Synthesis". For more information, please see our [project page](https://s-tian.github.io/projects/vista/).

## Setup
First, create a directory to act as a workspace. I would also recommend creating a virtual environment at this point (e.g. via conda or mamba). This code is tested with Python 3.8.

Then, clone the ZeroNVS repo (https://github.com/s-tian/ZeroNVS) into your workspace, and follow the setup instructions in that repository's README, including pulling the zeronvs_diffusion submodule. 

In the `ZeroNVS` directory, install the threestudio module:
```
cd threestudio
pip install -e .
cd ..
```

Clone this repo (https://github.com/s-tian/vista) into your workspace. Install the `robomimic`, `robosuite`, and `vipl` package as locally editable (note: robosuite and robomimic are also available on pip, but you need to install the local versions, as I've made edits to them):

```
cd robosuite
pip install -e .
cd ..
cd robomimic 
pip install -e .
cd ..
cd vipl
pip install -e .
```

Next, install `mimicgen_environments` and `robosuite_task_zoo`, by navigating to your workspace directory and then doing:
```
git clone https://github.com/NVlabs/mimicgen_environments
cd mimicgen_environments
pip install -e .
cd ..
git clone https://github.com/ARISE-Initiative/robosuite-task-zoo
git checkout 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed
cd robosuite_task_zoo
pip install -e .
```

Lastly, downgrade mujoco as instructed in the `mimicgen_environments` README. 

`pip install mujoco==2.3.2`

I've dumped the output of `pip list` and `mamba list` for working environment to this repo at `pip_list.txt` and `mamba_list.txt` for cross-reference. 

Note that this environment is not used for the reprojection baseline due to conflicts with Pytorch3D. To create that environment, follow the same steps but do not install ZeroNVS, instead install `pytorch3d==0.7.4`.

## Pretrained models
Pretrained ZeroNVS models are available on HuggingFace here: https://huggingface.co/s-tian/VISTA_Data/tree/main/models.
To use them, download the models and config file, and then set the constants in `vipl/vipl/utils/constants.py` to point to the downloaded models and config on your local machine. 

## Running experiments
The general workflow of running simulated experiments is:
1. Download source datasets for the task you want to run.
2. Convert the source dataset to a dataset containing potentially augmented observations.
3. Train a model on the potentially augmented dataset, and evaluate the model on held-out views of the target task.

### 1. Downloading source datasets
The source datasets for this project are directly obtained from robomimic and mimicgen_environments. The source datasets are converted to augmented datasets using the models, e.g. zeronvs, offline. Each source dataset was originally obtained from the download scripts in robomimic or mimicgen_environments. 

For convenience, we have uploaded the source datasets to HuggingFace here: https://huggingface.co/datasets/s-tian/VISTA_Data.

### 2. Converting source datasets to potentially augmented datasets
Conversion can be done by running the dataset_states_to_obs_zeronvs.py script in the provided robomimic package. For instance, let's look at the experiment in `experiments/paper/exp_1/small_perturb_lift_zeronvs`:

First, we run 

```
python dataset_states_to_obs_zeronvs.py --dataset ../../../mimicgen_environments/datasets/vista/exp_1/small_perturb/lift/low_dim_v141.hdf5 --output_name random_cam_zeronvs.hdf5  --done_mode 2 --randomize_cam_range small_perturb --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --compress --exclude-next-obs --randomize_cam --parse-iters 1 --camera_randomization_type zeronvs_lpips_guard 
```

which uses the zeronvs_lpips_guard camera randomization method to augment the low_dim_v141.hdf5 dataset we downloaded in step 1 (you may need to change the dataset path to match your local setup). 

Note that in `experiments/paper/exp_1/small_perturb_lift_zeronvs`, we break up this job into 10 commands to parallelize the process. When parallelizing, we provide a utility script `merge_hdf5.py` to merge the sliced hdf5 files after they are generated. 

### 3. Training a policy on the (potentially augmented) dataset

Continuing with the example in `experiments/paper/exp_1/small_perturb_lift_zeronvs`, we run 

```
python robomimic/robomimic/scripts/train.py --config robomimic/robomimic/exps/vista/exp_1/small_perturb/lift/zeronvs.json
```
In the config file `robomimic/robomimic/exps/vista/exp_1/small_perturb/lift/nvs.json`, you'll see the dataset to train on is specified as 

```
"data": "../mimicgen_environments/datasets/vista/exp_1/small_perturb/lift/random_cam_zeronvs.hdf5"
```

which you need to modify to point to the output of the previous step, which is by default generated in the same directory as your source dataset.

The training script uses robomimic conventions in general, but one detail to note is this entry:
```
    "random_camera_params": {
            "camera_name": "agentview",
            "sampler_type": "small_perturb"
        },
```
which specifies that a test environment will be created, with observations from the agentview camera randomized according to the `small_perturb` sampler type. This can be modified to test on other distributions of views (see `vipl/vipl/utils/camera_pose_sampler.py` for more details). 



