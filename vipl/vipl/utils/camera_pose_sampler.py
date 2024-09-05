import copy
import numpy as np
from scipy.spatial.transform import Rotation as R


def random_camera_perturbation(angle_std=0.1):
    # Generate random perturbation in orientation
    perturbation_axis = np.random.randn(3)
    perturbation_axis /= np.linalg.norm(perturbation_axis)
    perturbation_angle = np.random.normal(scale=angle_std)
    perturbation_quaternion = R.from_rotvec(perturbation_angle * perturbation_axis).as_quat()

    return perturbation_quaternion

def generate_random_camera_pose(current_pose, angle_std, position_std, num_samples=10):
    """
    :param current_pose: tuple of (position, orientation)
    :param angle_std:
    :param position_std:
    :param num_samples:
    :return:
    """
    current_position, current_orientation = current_pose
    generated_poses = []

    for _ in range(num_samples):
        # Generate random perturbation in orientation
        perturbation_quaternion = random_camera_perturbation(angle_std)

        # Also generate random perturbation in position
        perturbation_position = np.random.normal(scale=position_std, size=3)

        # Apply the perturbation to the current orientation
        new_orientation = R.from_quat(perturbation_quaternion) * R.from_quat(current_orientation)
        new_orientation = new_orientation.as_quat()

        new_pose = current_position + perturbation_position, new_orientation
        generated_poses.append(new_pose)

    return generated_poses


def get_world_direction(camera_extrinsics):
    """
    Given a quaternion representing the orientation of the camera in world frame where the z axis is along the viewing direction, 
    output the direction in the world frame
    that the camera is facing.
    """
    camera_orientation = R.from_quat(camera_extrinsics[1]).as_matrix()
    camera_axis_correction = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    )
    camera_orientation = camera_orientation @ camera_axis_correction
    camera_orientation = R.from_matrix(camera_orientation)
    camera_direction = camera_orientation.apply([0, 0, 1])
    return camera_direction


def generate_random_camera_pose_on_sphere(current_pose, radius, radius_std=0.05, num_samples=10, theta_range=None, phi_range=None):
    """
    Generate a random camera pose on the sphere with radius r and centered at a point radius away from the current pose along
    the viewing direction of the current pose.
    """
    # first compute the center of the sphere
    current_camera_position = current_pose[0]
    generated_poses = []
    for i in range(num_samples):
        radius_sample = radius + np.random.normal(scale=radius_std)
        sphere_center = current_camera_position + radius_sample * get_world_direction(current_pose)
        # Generate a random point on the sphere that is centered at the sphere center
        # and has radius r, and points at the sphere center, with the same z height as current_camera_position
        if theta_range is None:
            theta_range = [0, 2 * np.pi]
        if phi_range is None:
            phi_range = [0, np.pi]
        theta = np.random.uniform(*theta_range)  # Azimuthal angle
        phi = np.arccos((current_camera_position[-1] - sphere_center[-1]) / radius_sample)

        # Convert spherical coordinates to Cartesian coordinates
        x = sphere_center[0] + radius_sample * np.sin(phi) * np.cos(theta)
        y = sphere_center[1] + radius_sample * np.sin(phi) * np.sin(theta)
        z = current_camera_position[-1] 

        random_point_on_sphere = np.array([x, y, z])

        # Create a camera pose that has the given position and looks at the sphere center
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = random_point_on_sphere
        camera_pose[:3, :3] = np.eye(3)
        camera_pose[:3, 2] = sphere_center - random_point_on_sphere
        camera_pose[:3, 2] /= np.linalg.norm(camera_pose[:3, 2])

        # Flip the up vector to fix the upside-down rendering issue
        camera_pose[:3, 1] = -camera_pose[:3, 1]
        camera_pose[:3, 2] = -camera_pose[:3, 2]

        camera_pose[:3, 0] = np.cross([0, 0, 1], camera_pose[:3, 2])
        camera_pose[:3, 0] /= np.linalg.norm(camera_pose[:3, 0])
        camera_pose[:3, 1] = np.cross(camera_pose[:3, 2], camera_pose[:3, 0])
        camera_pose[:3, 1] /= np.linalg.norm(camera_pose[:3, 1])

        # Convert the camera pose to position and orientation matrix
        position_matrix = camera_pose[:3, 3]
        orientation_matrix = camera_pose[:3, :3]
        # Convert the orientation to a xyzw quaternion
        orientation_quaternion = R.from_matrix(orientation_matrix).as_quat()
        generated_poses.append((position_matrix, orientation_quaternion))
    return generated_poses


class CameraPoseSampler:

    SMALL_PERTURB_ANGLE_STD = 0.075
    SMALL_PERTURB_POSITION_STD = 0.03

    ARC_90DEG_RADIUS_STD = 0.05
    ARC_90DEG_ANGLE_RANGE = np.pi/4

    def __init__(self, sampler_type):
        self.sampler_type = sampler_type
        assert self.sampler_type in ["small_perturb", "arc_90deg", "small_perturb_real"], "Sampler not implemented!"
        self.robot_base_pos = np.array([0.01375589, 0.0, 0.83170968]) # harvested from Lift environment, set y = 0 to center in left/right directions. 

    def sample_poses(self, n, starting_pose):
        """
        Sample camera poses based on the sampler type specified during initialization.

        Args:
            n (int): Number of camera poses to generate.
            starting_pose (tuple): A tuple containing the starting position and orientation of the camera.
                                   The position should be a numpy array of shape (3,), and the orientation
                                   should be a quaternion numpy array of shape (4,).

        Returns:
            list: A list of tuples, where each tuple contains the position and orientation of the camera.
                  The position is a numpy array of shape (3,), and the orientation is a quaternion numpy
                  array of shape (4,).
        """
        initial_camera_pose = copy.deepcopy(starting_pose)
        if self.sampler_type == "small_perturb":
            return generate_random_camera_pose(initial_camera_pose, angle_std=self.SMALL_PERTURB_ANGLE_STD, position_std=self.SMALL_PERTURB_POSITION_STD, num_samples=n)
        elif self.sampler_type == "arc_90deg":
            radius = np.linalg.norm(initial_camera_pose[0] - self.robot_base_pos)
            return generate_random_camera_pose_on_sphere(
                current_pose=initial_camera_pose, 
                radius=radius, 
                radius_std=self.ARC_90DEG_RADIUS_STD, 
                theta_range=[-1 * self.ARC_90DEG_ANGLE_RANGE, self.ARC_90DEG_ANGLE_RANGE],
                num_samples=n
                )
        elif self.sampler_type == "small_perturb_real":
            return generate_random_camera_pose(initial_camera_pose, angle_std=self.SMALL_PERTURB_ANGLE_STD*5, position_std=self.SMALL_PERTURB_POSITION_STD*5, num_samples=n)