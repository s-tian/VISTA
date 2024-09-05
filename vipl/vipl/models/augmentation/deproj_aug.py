import torch
import math
import numpy as np
from PIL import Image
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
)
from einops import rearrange
from pytorch3d.renderer.points.compositor import _add_background_color_to_images
from pytorch3d.structures import Pointclouds
from kornia.geometry import PinholeCamera
from diffusers import StableDiffusionInpaintPipeline


from vipl.models.augmentation.base_aug import BaseAugModel
from vipl.utils.cam_utils import opencv2opengl
from typing import List, Optional, Tuple, Union



def opengl_to_pytorch3d(cam2world_opengl):
    # Copy the original matrix to avoid modifying it directly
    cam2world_pytorch3d = cam2world_opengl.copy()
    cam2world_pytorch3d[:, 2] *= -1
    # In OpenGL, x is right, so we need to negate x axis to make it left in PyTorch3D convention
    cam2world_pytorch3d[:, 0] *= -1
    return cam2world_pytorch3d


BG_COLOR=(1.0, 1.0, 1.0)

class PointsRenderer(torch.nn.Module):
    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def forward(self, point_clouds, return_z=False, return_bg_mask=False, return_fragment_idx=False,
                **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)
        r = self.rasterizer.raster_settings.radius

        zbuf = fragments.zbuf.permute(0, 3, 1, 2)
        fragment_idx = fragments.idx.long().permute(0, 3, 1, 2)
        background_mask = fragment_idx[:, 0] < 0  # [B, H, W]
        images = self.compositor(
            fragment_idx,
            zbuf,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        ret = [images]
        if return_z:
            ret.append(fragments.zbuf)
        if return_bg_mask:
            ret.append(background_mask)
        if return_fragment_idx:
            ret.append(fragments.idx.long())

        if len(ret) == 1:
            ret = images
        return ret



class SoftmaxImportanceCompositor(torch.nn.Module):
    """
    Accumulate points using a softmax importance weighted sum.
    """

    def __init__(
        self, background_color: Optional[Union[Tuple, List, torch.Tensor]] = None, softmax_scale=1.0,
    ) -> None:
        super().__init__()
        self.background_color = background_color
        self.scale = softmax_scale

    def forward(self, fragments, zbuf, ptclds, **kwargs) -> torch.Tensor:
        """
        Composite features within a z-buffer using importance sum. Given a z-buffer
        with corresponding features and weights, these values are accumulated
        according to softmax(1/z * scale) to produce a final image.

        Args:
            fragments: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
                giving the indices of the nearest points at each pixel, sorted in z-order.
                Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
                feature of the kth closest point (along the z-direction) to pixel (y, x) in
                batch element n.
            zbuf: float32 Tensor of shape (N, points_per_pixel, image_size,
                image_size) giving the depth value of each point in the z-buffer.
                Value -1 means no points assigned to the pixel.
            pt_clds: Packed feature tensor of shape (C, P) giving the features of each point
                (can use RGB for example).

        Returns:
            images: Tensor of shape (N, C, image_size, image_size)
                giving the accumulated features at each point.
        """
        background_color = kwargs.get("background_color", self.background_color)

        zbuf_processed = zbuf.clone()
        zbuf_processed[zbuf_processed < 0] = - 1e-4
        importance = 1.0 / (zbuf_processed + 1e-6)
        weights = torch.softmax(importance * self.scale, dim=1)

        fragments_flat = fragments.flatten()
        gathered = ptclds[:, fragments_flat]
        gathered_features = gathered.reshape(ptclds.shape[0], fragments.shape[0], fragments.shape[1], fragments.shape[2], fragments.shape[3])
        images = (weights[None, ...] * gathered_features).sum(dim=2).permute(1, 0, 2, 3)

        # images are of shape (N, C, H, W)
        # check for background color & feature size C (C=4 indicates rgba)
        if background_color is not None:
            return _add_background_color_to_images(fragments, images, background_color)
        return images

class DeprojModel(BaseAugModel):

    """
    Implements a baseline that deprojects the original image to a 3D point cloud, then reprojects it to the target camera.
    The depth is computed using an off the shelf depth estimation model.
    """

    def __init__(self, depth_range=(0.55046624, 2.7143734), depth=None, model="zoe", device="cuda"):
        print("\nLoading deprojection model...\n")
        self.device = device
        if depth is not None:
            print(" !!!!!!!!!!!!!!!!!!!! Debugging mode -- using ground truth depth !!!!!!!!!!!!!!!!!!!!") 
            self.depth = np.squeeze(depth)
            self.depth_debug = True
        elif model == "midas":
            print("\nLoading MiDaS depth estimator...\n")
            self.midas_depth = torch.hub.load("intel-isl/MiDaS", "DPT_BEiT_L_512").to(device)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.midas_transform = midas_transforms.beit512_transform
            self.midas_depth.eval()
            self.depth_debug = False
        elif model == "zoe":
            print("\n Loading ZoeDepth depth estimator...")
            self.model_zoe_nk = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).to(device)
            self.model_zoe_nk.eval()

        self.model = model
        self.depth_range = depth_range
        self.depth_shift = 0 
        self.fg_depth_range = 0.0015
        self.background_hard_depth = self.depth_shift + self.fg_depth_range
        self.background_hard_depth = 1
        self.point_size_min_ratio = 1
        self.point_size = 0.0005
        self.sky_point_size_multiplier = 1.5

        print("\nLoading StableDiffusion2 Inpainting...\n")
        # inpainting
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False
        )
        self.pipe.to(self.device)

    def get_depth(self, image):
        """
        image is a PIL image
        return a tuple: (depth, disparity) both as torch.Tensors
        """
        if self.model == "midas":
            return self.get_depth_midas(image)
        elif self.model == "zoe":
            return self.get_depth_zoe(image)
        else:
            raise ValueError(f"Depth model name {self.model} not implemented")

    @torch.no_grad()
    def get_depth_zoe(self, image):
        depth_tensor = self.model_zoe_nk.infer_pil(image, output_type="tensor")[None, None].to(self.device)
        disparity = 1 / depth_tensor
        return depth_tensor, disparity

    @torch.no_grad()
    def get_depth_midas(self, image):
        # Image is a PIL image
        if self.depth_debug:
            depth = torch.tensor(self.depth[None, None], device=self.device) 
            disparity = 1.0 / depth
            return depth, disparity
            
        image = np.array(image)
        disparity = self.midas_depth(self.midas_transform(image).to(self.device))
        disparity = torch.nn.functional.interpolate(
            disparity.unsqueeze(1),
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        disparity = disparity.clip(1e-6, max=None)
        depth = 1 / disparity
        if self.depth_range is not None:
            depth_min, depth_max = self.depth_range
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * (depth_max - depth_min) + depth_min
        disparity = 1 / depth
        return depth, disparity

    def get_init_camera(self):
        focal_length = 0.5 * 256 / math.tan(45 * math.pi / 360)
        K = torch.zeros((1, 4, 4), device=self.device)
        K[0, 0, 0] = focal_length
        K[0, 1, 1] = focal_length
        K[0, 0, 2] = 256 // 2
        K[0, 1, 2] = 256 // 2
        K[0, 2, 3] = 1
        K[0, 3, 2] = 1
        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = torch.zeros((1, 3), device=self.device)
        camera = PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=((256, 256),), device=self.device)
        return camera

    def get_pytorch_3d_camera(self, extrinsics, image_size=256):
        # extrinsics is a world2cam matrix.
        # pytorch3d assumes row vectors
        extrinsics = extrinsics.T
        extrinsics = torch.clone(torch.tensor(extrinsics, device=self.device).to(torch.float32))
        fovy = 45
        focal_length = 0.5 * image_size / math.tan(fovy * math.pi / 360)
        K = torch.zeros((1, 4, 4), device=self.device)
        K[0, 0, 0] = focal_length
        K[0, 1, 1] = focal_length
        K[0, 0, 2] = image_size // 2
        K[0, 1, 2] = image_size // 2
        K[0, 2, 3] = 1
        K[0, 3, 2] = 1
        R = extrinsics[:3, :3]        
        T = extrinsics[3, :3]
        R, T = R[None], T[None]
        camera = PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=((image_size, image_size),), device=self.device)
        return camera


    def get_camera_frame_points(self, size, device="cuda"):
        x = torch.arange(size).float() + 0.5
        y = torch.arange(size).float() + 0.5
        points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
        # (h w c) -> (h w) c
        points = points.reshape(-1, 2).to(device)
        return points

    def augment(self, original_image, target_camera, original_camera=None, convention="opengl"):
        """
        :param original_image: A PIL image of the original image
        :param target_camera: Target camera rotation matrix (cam2world)
        :param original_camera: (optional) original camera rotation matrix. (cam2world)
        If None, assume the identity matrix.
        :return: A PIL image (hopefully) rendered from the target camera, of the scene from original_image.
        """
        if original_camera is None:
            original_camera = np.eye(4)

        # convert cam2world matrix into opengl
        original_camera = self.convert_cam2world_to_opengl(original_camera, convention)
        target_camera = self.convert_cam2world_to_opengl(target_camera, convention)

        # now original and target camera are both cam2world matrices in opengl convention 
        # print("Opengl format target cam:", target_camera)
        depth, disparity = self.get_depth(original_image)
        original_image_tensor = torch.from_numpy(np.array(original_image) / 255.).to(self.device)
        # get the camera matrix
        target_camera_py3d = self.get_pytorch_3d_camera(np.linalg.inv(opengl_to_pytorch3d(target_camera)))
        original_camera_matrix = convert_pytorch3d_kornia(self.get_pytorch_3d_camera(np.linalg.inv(opengl_to_pytorch3d(original_camera))))
        camera_frame_points = self.get_camera_frame_points(depth.shape[-1], device=self.device)
        # convert depth from (b, c, h, w) to (w h b), c
        point_depth = rearrange(depth, 'b c h w -> (w h b) c')
        # original_image_tensor is (h, w, c) already
        colors = rearrange(original_image_tensor, 'h w c -> (w h) c')
        points_3d = original_camera_matrix.unproject(camera_frame_points, point_depth)
        
        depth_normalizer = self.background_hard_depth
        min_ratio = self.point_size_min_ratio
        radius = self.point_size * (
                    min_ratio + (1 - min_ratio) * (point_depth.permute([1, 0]) / depth_normalizer))
        radius = radius.clamp(max=self.point_size * self.sky_point_size_multiplier)
        raster_settings = PointsRasterizationSettings(
            image_size=256,
            radius=0.007, # this works well for small perturbations on robosuite tasks
            points_per_pixel=8,
        )
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=target_camera_py3d, raster_settings=raster_settings),
            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
        )
        point_cloud = Pointclouds(points=[points_3d], features=[colors])
        images, zbuf, bg_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)
        novel_view_pil = Image.fromarray(np.clip(images[0].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8))
        # The mask structure is white for inpainting and black for keeping as is
        # convert from a boolean array of (h, w) to a (h, w, 3) array of 0s and 255s
        bg_mask = bg_mask[0].detach().cpu().numpy()
        bg_mask = np.stack([bg_mask * 255] * 3, axis=-1).astype(np.uint8)
        bg_mask = Image.fromarray(bg_mask)
        novel_view_pil = self.inpaint(novel_view_pil, bg_mask, "A simulated scene with a white table surface and some objects on top of it, high quality, render")

        return novel_view_pil

    def inpaint(self, image, mask_image, prompt):
        # if the PIL images are not 512x512, save the original sizes
        image_size = image.size
        image = image.resize((512, 512))
        mask_image = mask_image.resize((512, 512))
        # make sure all values in the mask are 0 or 255
        mask_image = mask_image.point(lambda p: p > 128 and 255)
        mask_image.save("mask_image_512.png")
        image = np.array(image) / 255.
        mask_image = np.array(mask_image)[..., 0] / 255.
        output_image = self.pipe(prompt=prompt, 
                                 image=image, 
                                 mask_image=mask_image,
                                 negative_prompt="noise,static,stripes,lines").images[0]
        output_image = output_image.resize(image_size)
        return output_image



def get_transformation_between(view1, view2):
    return np.linalg.inv(view1) @ view2


def convert_pytorch3d_kornia(camera, size=256, fov=45):
    # Assume a square image
    # Note that fov is in degrees
    # size is the resolution of the image in pixels
    focal_length = 0.5 * size / math.tan(fov * math.pi / 360)
    transform_matrix_pt3d = camera.get_world_to_view_transform().get_matrix()[0]
    transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)

    pt3d_to_kornia = torch.diag(torch.tensor([-1., -1, 1, 1], device=camera.device))
    transform_matrix_w2c_kornia = pt3d_to_kornia @ transform_matrix_w2c_pt3d

    extrinsics = transform_matrix_w2c_kornia.unsqueeze(0)
    h = torch.tensor([size], device="cuda")
    w = torch.tensor([size], device="cuda")
    K = torch.eye(4)[None].to("cuda")
    K[0, 0, 2] = size // 2
    K[0, 1, 2] = size // 2
    K[0, 0, 0] = focal_length
    K[0, 1, 1] = focal_length
    return PinholeCamera(K, extrinsics, h, w)
