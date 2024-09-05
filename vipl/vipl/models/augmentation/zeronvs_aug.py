import copy
import lpips
import tqdm
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import torch
from robomimic.envs.env_robosuite import EnvRobosuite
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import CameraMover, generate_random_camera_pose

from ldm.models.diffusion import options

options.LDM_DISTILLATION_ONLY = True

from threestudio.models.guidance import zero123_guidance
from omegaconf import OmegaConf

from vipl.models.augmentation.base_aug import BaseAugModel
from vipl.utils.cam_utils import opencv2opengl
from vipl.utils.exp_utils import compute_lpips


class ZeroNVSModel(BaseAugModel):

    def __init__(self, checkpoint, config, zeronvs_params, debug_resize=False, device="cuda"):
        self.device = device
        self.checkpoint = checkpoint
        self.config = config
        self.debug_resize = debug_resize
        self.guidance = self._setup_zeronvs_model(pretrained_checkpoint=checkpoint, pretrained_config=config)
        self.zeronvs_params = dict()
        # defaults
        self.zeronvs_params["ddim_steps"] = 250 
        self.zeronvs_params["ddim_eta"] = 1.0
        self.zeronvs_params["lpips_loss_threshold"] = 0.9
        self.zeronvs_params["num_tries"] = 5
        self.zeronvs_params["fov_deg"] = 45
        # defaults
        self.zeronvs_params.update(zeronvs_params) # use passed args to override zeronvs default params
        print("ZeroNVS precomputed scale: ", self.zeronvs_params["precomputed_scale"])
        print("ZeroNVS FOV Deg: ", self.zeronvs_params["fov_deg"])
        print("LPIPS loss threshold:", self.zeronvs_params["lpips_loss_threshold"])
        self.lpips = lpips.LPIPS(net="alex").to(device)

    @torch.no_grad()
    def augment(self, original_image, target_camera, original_camera=None, convention="opengl"):
        """
        :param original_image: A PIL image of the original image
        :param target_camera: Target camera rotation matrix (cam2world)
        :param original_camera: (optional) original camera rotation matrix. (cam2world)
        If None, assume the identity matrix.
        :return: A PIL image (hopefully) rendered from the target camera, of the scene from original_image.
        """

        if self.debug_resize:
            original_image = original_image.resize((84, 84))
            original_image = original_image.resize((256, 256))

        if original_camera is None:
            original_camera = np.eye(4)

        # convert cam2world matrix into opengl
        original_camera = self.convert_cam2world_to_opengl(original_camera, convention)
        target_camera = self.convert_cam2world_to_opengl(target_camera, convention)

        lpips_loss = 1
        num_tries = 0
        while lpips_loss > self.zeronvs_params["lpips_loss_threshold"] and num_tries < self.zeronvs_params["num_tries"]:
            next_obs_augmented = self._perform_nvs(
                guidance=self.guidance,
                original_image=original_image,
                original_camera=original_camera,
                target_camera=target_camera,
                ddim_steps=self.zeronvs_params["ddim_steps"],
                ddim_eta=self.zeronvs_params["ddim_eta"],
                scene_scale=self.zeronvs_params["precomputed_scale"],
            )
            lpips_loss = compute_lpips(
                lpips=self.lpips,
                image1=original_image,
                image2=next_obs_augmented,
            )
            print("LPIPS is: ", lpips_loss)
            num_tries += 1
        if lpips_loss < self.zeronvs_params["lpips_loss_threshold"]:
            return next_obs_augmented
        else:
            return original_image

    def _setup_zeronvs_model(self, pretrained_checkpoint, pretrained_config):
        guidance_cfg = dict(
            pretrained_model_name_or_path=pretrained_checkpoint,
            pretrained_config=pretrained_config,
            guidance_scale=7.5,
            cond_image_path="/viscam/projects/vipl/nextnvs/motorcycle.png",  # unused
            min_step_percent=[0, .75, .02, 1000],
            max_step_percent=[1000, 0.98, 0.025, 2500],
            vram_O=False
        )
        guidance = zero123_guidance.Zero123Guidance(OmegaConf.create(guidance_cfg))
        return guidance

    def _perform_nvs(self, guidance, original_image, original_camera, target_camera, scene_scale=0.9, guidance_scale=7.5, ddim_steps=250, ddim_eta=1):
        assert isinstance(original_image, Image.Image), "original_image must be a PIL Image"
        cond_image_pil = original_image
        cond_image_pil = cond_image_pil.resize((256, 256))
        cond_image = torch.from_numpy(np.array(cond_image_pil)).cuda() / 255.
        c_crossattn, c_concat = guidance.get_img_embeds(
            cond_image.permute((2, 0, 1))[None])

        target_camera = torch.from_numpy(target_camera[None]).cuda().to(torch.float32)
        cond_camera = torch.from_numpy(original_camera[None]).cuda().to(torch.float32)

        fov_deg = self.zeronvs_params["fov_deg"]

        camera_batch = {
            "target_cam2world": target_camera,
            "cond_cam2world": cond_camera,
            "fov_deg": torch.from_numpy(np.array([fov_deg])).cuda().to(torch.float32)
        }

        guidance.cfg.precomputed_scale = scene_scale
        cond = guidance.get_cond_from_known_camera(
            camera_batch,
            c_crossattn=c_crossattn,
            c_concat=c_concat,
        )
        novel_view = guidance.gen_from_cond(cond, scale=guidance_scale, ddim_steps=ddim_steps, ddim_eta=ddim_eta)  # play with eta, try increasing guidance
        novel_view_pil = Image.fromarray(np.clip(novel_view[0] * 255, 0, 255).astype(np.uint8))
        return novel_view_pil