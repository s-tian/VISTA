from vipl.utils.constants import ZERONVS_CONFIG_PATH, ZERONVS_CHECKPOINT_PATH, ZERONVS_MIMICGEN_PATH, ZERONVS_DROID_PATH

def get_model_by_name(name, **kwargs):
    # TODO Fix kwargs for zeronvs (the way the dict and the kwargs to this function interact is unintuitive...)
    if name == "deproj":
        from vipl.models.augmentation.deproj_aug import DeprojModel
        return DeprojModel(**kwargs)
    elif name == "zeronvs":
        input("You are using the zeronvs without LPIPS guard version. It's recommended to use the LLPIPS guard variant. Please verify: ")
        from vipl.models.augmentation.zeronvs_aug import ZeroNVSModel
        precomputed_scale = 0.6
        return ZeroNVSModel(
            checkpoint=ZERONVS_CHECKPOINT_PATH,
            config=ZERONVS_CONFIG_PATH,
            zeronvs_params=dict(
                ddim_steps=250,
                ddim_eta=1.0,
                precomputed_scale=precomputed_scale,
            ),
            **kwargs
        )
    elif name == "zeronvs_lpips_guard":
        from vipl.models.augmentation.zeronvs_aug import ZeroNVSModel
        precomputed_scale = 0.6
        return ZeroNVSModel(
            checkpoint=ZERONVS_CHECKPOINT_PATH,
            config=ZERONVS_CONFIG_PATH,
            zeronvs_params=dict(
                ddim_steps=250,
                ddim_eta=1.0,
                precomputed_scale=precomputed_scale,
                lpips_loss_threshold=0.5,
            ),
            **kwargs
        )
    elif name == "zeronvs_lpips_guard_real":
        from vipl.models.augmentation.zeronvs_aug import ZeroNVSModel
        precomputed_scale = 0.6
        return ZeroNVSModel(
            checkpoint=ZERONVS_CHECKPOINT_PATH,
            config=ZERONVS_CONFIG_PATH,
            zeronvs_params=dict(
                ddim_steps=250,
                ddim_eta=1.0,
                precomputed_scale=precomputed_scale,
                lpips_loss_threshold=0.7,
                fov_deg=70,
            ),
            **kwargs
        )
    elif name == "zeronvs_ft":
        ft_path = ZERONVS_DROID_PATH
        from vipl.models.augmentation.zeronvs_aug import ZeroNVSModel
        return ZeroNVSModel(
            checkpoint=ft_path,
            config=ZERONVS_CONFIG_PATH,
            zeronvs_params=dict(
                ddim_steps=250,
                ddim_eta=1.0,
                precomputed_scale=0.6,
                fov_deg=70,
                lpips_loss_threshold=0.7,
            ),
            **kwargs
        )
    elif name == "zeronvs_mimicgen_ft":
        from vipl.models.augmentation.zeronvs_aug import ZeroNVSModel
        ft_path = ZERONVS_MIMICGEN_PATH
        precomputed_scale = 0.6
        return ZeroNVSModel(
            checkpoint=ft_path,
            config=ZERONVS_CONFIG_PATH,
            zeronvs_params=dict(
                ddim_steps=250,
                ddim_eta=1.0,
                precomputed_scale=precomputed_scale,
                lpips_loss_threshold=0.5,
            ),
            **kwargs
        )
    elif name == "clone":
        from vipl.models.augmentation.dummy_aug import DummyAugModel
        return DummyAugModel()
    else:
        raise ValueError(f"Invalid model name")
