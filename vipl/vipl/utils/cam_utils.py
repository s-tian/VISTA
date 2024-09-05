import copy
import numpy as np
from scipy.spatial.transform import Rotation as R

import robosuite.utils.transform_utils as T

def posori_to_rotmat(position, orientation):
    # given the position and orientation of the camera as a quaternion, return the cam2world matrix
    # position is a 3-tuple, orientation is a 4-tuple
    return T.make_pose(position, T.quat2mat(orientation))

def droid_extrinsics_to_cam2world(extrinsics):
    #  => Extrinsics are saved as a 6-dim vector of [pos; rot] where:
    #       - `pos` is (x, y, z) offset --> moving left of robot is +y, moving right is -y
    #       - `rot` is rotation offset as Euler (`R.from_matrix(rmat).as_euler("xyz")`)
    # convert to a 4x4 cam2world matrix (opencv convention)
    cam_rot = R.from_euler('xyz', extrinsics[3:])
    cam_rot = cam_rot.as_matrix()
    cam_pos = extrinsics[:3]
    cam_extrinsics = np.eye(4)
    cam_extrinsics[:3, :3] = cam_rot
    cam_extrinsics[:3, 3] = cam_pos
    return cam_extrinsics

def cam2world_2_droid_extrinsics(cam2world):
    cam_rot = cam2world[:3, :3]
    cam_pos = cam2world[:3, 3]
    cam_rot = R.from_matrix(cam_rot)
    cam_rot = cam_rot.as_euler('xyz')
    return np.concatenate([cam_pos, cam_rot])

def cam2world_2_posori(cam2world):
    return T.mat2pose(cam2world)

def opencv2opengl(cam2world):
    opengl_c2w = copy.deepcopy(cam2world)
    opengl_c2w[:, 1] *= -1
    opengl_c2w[:, 2] *= -1
    return opengl_c2w

 