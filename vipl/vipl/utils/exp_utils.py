import torch
import numpy as np

model_to_label = {
    "clone": "Clone",
    "zeronvs": "ZeroNVS",
    "zeronvs_lpips_guard": "ZeroNVS (LPIPS limit)",
    "deproj": "Reproj+Inpaint",
    "zeronvs_ft": "FT on DROID",
    "zeronvs_mimicgen_ft": "FT on MimicGen",
}

def compute_lpips(lpips, image1, image2):
    # compute the lpips between two images as PIL images
    image1 = torch.from_numpy(np.array(image1)).cuda() / 255.
    image2 = torch.from_numpy(np.array(image2)).cuda() / 255.
    image1 = image1.permute((2, 0, 1))[None]
    image2 = image2.permute((2, 0, 1))[None]
    return lpips(image1, image2).item()
