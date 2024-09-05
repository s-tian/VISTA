"""
Given a list of HDF5 files, merges them into a single HDF5 file. The format of the HDF5 files is as follows:

- `data` (group)

  - `total` (attribute) - number of state-action samples in the dataset

  - `env_args` (attribute) - a json string that contains metadata on the environment and relevant arguments used for collecting data

  - `demo_0` (group) - group for the first demonstration (every demonstration has a group)

    - `num_samples` (attribute) - the number of state-action samples in this trajectory
    - `model_file` (attribute) - the xml string corresponding to the MJCF MuJoCo model
    - `states` (dataset) - flattened raw MuJoCo states, ordered by time
    - `actions` (dataset) - environment actions, ordered by time

  - `demo_1` (group) - group for the second demonstration

    ...

Where each dataset will contain a different subset of the demos.
"""

import os
import argparse
import h5py
import glob

def main(args):
    f_out = h5py.File(args.output_file, "w")
    data_grp = f_out.create_group("data")

    # merge all the input files
    # glob all files that match the input pattern and are not the output file
    all_files = []
    output_file_abs_path = os.path.abspath(args.output_file)
    for i, f_in in enumerate(args.input_files):
        matched_files = glob.glob(f_in)
        all_files.extend([f for f in matched_files if os.path.abspath(f) != output_file_abs_path])
    print(all_files)
    total_samples = 0
    for i, f_in in enumerate(all_files):
        print("Processing file {} of {}...".format(i + 1, len(all_files)))
        print(f_in)
        f_in = h5py.File(f_in, "r")
        data_grp.attrs["total"] = total_samples + f_in["data"].attrs["total"]
        data_grp.attrs["env_args"] = f_in["data"].attrs["env_args"]
        # get all the demos in this file, which may not start with 0
        demos = [int(k.split("_")[1]) for k in f_in["data"].keys() if k.startswith("demo")]
        # sort demos by index
        demos = sorted(demos)
        for demo_idx in demos:
            demo_grp_key = "data/demo_{}".format(demo_idx)
            demo_grp = f_in[demo_grp_key]
            if demo_grp_key in data_grp:
                print(f"{demo_grp_key} already exists in file, overwriting!")
            # create a group for this demo
            # demo_grp_out = data_grp.create_group("demo_{}".format(demo_idx))
            # clone the entire demo group
            demo_grp.copy(demo_grp, data_grp)
        # copy mask
        if i == 0:
            f_in.copy(f_in["mask"], f_out)
        f_in.close()
    f_out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="List of input HDF5 files to merge",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output HDF5 file to write merged data to",
    )
    args = parser.parse_args()
    main(args)