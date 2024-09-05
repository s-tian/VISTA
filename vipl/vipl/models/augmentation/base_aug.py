import abc

from vipl.utils.cam_utils import opencv2opengl

class BaseAugModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def augment(self, original_image, target_camera, original_camera=None, convention="opengl"):
        """
        :param original_image: A PIL image of the original image
        :param target_camera: Target camera rotation matrix
        :param original_camera: (optional) original camera rotation matrix.
        If None, assume the identity matrix.
        :return: A PIL image (hopefully) rendered from the target camera, of the scene from original_image.
        """
        raise NotImplementedError

    def convert_cam2world_to_opengl(self, cam_matrix, convention):
        if convention == "opencv":
            return opencv2opengl(cam_matrix)
        elif convention == "opengl":
            return cam_matrix
        else:
            raise NotImplementedError(f"Camera coordinate convention {convention} not implemented.")