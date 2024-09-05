from vipl.models.augmentation.base_aug import BaseAugModel

class DummyAugModel(BaseAugModel):

    def augment(self, original_image, target_camera, original_camera=None, convention="opengl"):
        """
        :param original_image: A PIL image of the original image
        :param target_camera: Target camera rotation matrix
        :param original_camera: (optional) original camera rotation matrix.
        If None, assume the identity matrix.
        :return: A PIL image (hopefully) rendered from the target camera, of the scene from original_image.
        """
        return original_image