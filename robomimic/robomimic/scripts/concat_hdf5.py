import os
import argparse
import h5py
import numpy as np
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
    demo_counter = 0
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
        m = {}
        for demo_idx in demos:
            demo_grp_key = "demo_{}".format(demo_counter)
            demo_grp = f_in["data/demo_{}".format(demo_idx)]
            m[f"demo_{demo_idx}"] = f"demo_{demo_counter}" 
            if demo_grp_key in data_grp:
                print(f"{demo_grp_key} already exists in file, overwriting!")
            # create a group for this demo
            demo_grp_out = data_grp.create_group(demo_grp_key)
            # copy the contents of the demo group
            for key in demo_grp.keys():
                # if it's a group:
                if isinstance(demo_grp[key], h5py.Group):
                    demo_grp[key].copy(demo_grp[key], demo_grp_out)
                else:
                    # if it's a dataset, create it
                    demo_grp_out.create_dataset(key, data=demo_grp[key][()])
            # also copy all attrs to the new dataset
            for key in demo_grp.attrs:
                demo_grp_out.attrs[key] = demo_grp.attrs[key]
            demo_counter += 1
        # copy mask
        if i == 0:
            f_in.copy(f_in["mask"], f_out)
        else:
            # use m to construct a new mask
            for mask in f_in["mask"]:
                mask_data = f_in['mask'][mask]
                # replace each item in mask_data with m[item]
                new_mask = mask_data[()] 
                new_mask_2 = []
                for key in mask_data:
                    k2 = key.decode("utf-8")
                    new_mask_2.append(m[k2])
                # modify the mask data in f_out
                # concatenate
                new_mask = np.concatenate((new_mask, new_mask_2))
                # convert <U8 to |S8
                new_mask = new_mask.astype('|S8')
                del f_out['mask'][mask]
                f_out['mask'][mask] = new_mask

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