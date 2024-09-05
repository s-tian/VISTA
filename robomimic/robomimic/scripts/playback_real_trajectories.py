import argparse
import os
import h5py
import imageio


def main(args):
    video_writer = imageio.get_writer(args.output, fps=5, mode="I", loop=0)
    # given a directory containing many directories, look in each directory for args.filename and open it
    num_taken = 0
    for root, dirs, files in os.walk(args.directory):
        if args.filename in files:
            print(f"Found {args.filename} in {root}")
            if root.endswith("Tue_Jun__4_16:19:30_2024"):
                continue
                # open file as h5py file
            with h5py.File(os.path.join(root, args.filename), "r") as f:
                # get the image dataset
                images = f["observation/camera/image/varied_camera_1_left_image"][()]
                # for each image in the dataset, write it to the video writer
                for i, img in enumerate(images):
                    if i % args.skip == 0:
                        video_writer.append_data(img)
                num_taken += 1
                if num_taken == args.n:
                    video_writer.close()
                    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play back real robot trajectories")
    parser.add_argument("--directory", type=str, help="Directory to parse.")
    parser.add_argument("--filename", type=str, help="filename to look for .")
    parser.add_argument("--n", type=int, default=1, help="num trajectories to render")
    parser.add_argument("--skip", type=int, default=1, help="skip every n frames")
    parser.add_argument("--output", type=str, help="File to output to.")

    args = parser.parse_args()
    main(args)