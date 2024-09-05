"""
Given an HDF5 file, outputs a new HDF5 file containing only the first n demonstrations based on their demo_ numbers. The script will only copy fields that are within the 'data' group.
"""

import argparse
import h5py

def main(args):
    # Open the input HDF5 file
    f_in = h5py.File(args.input_file, "r")
    # Create the output HDF5 file
    f_out = h5py.File(args.output_file, "w")
    data_grp_in = f_in["data"]
    data_grp_out = f_out.create_group("data")

    # Copy attributes from the input 'data' group to the output 'data' group
    for attr in data_grp_in.attrs:
        data_grp_out.attrs[attr] = data_grp_in.attrs[attr]

    # Get all demo groups sorted by their indices
    demo_keys = sorted([key for key in data_grp_in.keys() if key.startswith("demo_")],
                       key=lambda x: int(x.split('_')[1]))

    # Copy the first n demos
    for demo_key in demo_keys[:args.num_demos]:
        data_grp_in.copy(demo_key, data_grp_out)

    # Close the files
    f_in.close()
    f_out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        required=True,
        help="Input HDF5 file to process",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output HDF5 file to write the selected data to",
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        required=True,
        help="Number of initial demonstrations to include in the output file",
    )
    args = parser.parse_args()
    main(args)
