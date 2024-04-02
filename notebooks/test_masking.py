from pathlib import Path

import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

import numpy as np
import torch as T
import torchvision.transforms as TV
from PIL import Image
from torchvision.utils import save_image

from src.datamodules.utils import beit_block_masking
from src.models2.utils import images_to_tokens, tokens_to_images

# Load an image and convert to a tensor
transforms = TV.Compose([
    TV.Resize(224),
    TV.CenterCrop(224),
    TV.ToTensor(),
])
image_folder = "/srv/beegfs/scratch/groups/rodem/datasets/ImageNet/val/n04118776/"
image_names = ["ILSVRC2012_val_00011948.JPEG", "ILSVRC2012_val_00001210.JPEG"]
images = [Image.open(Path(image_folder, n)) for n in image_names]
images = [transforms(i).unsqueeze(0) for i in images]
images = T.vstack(images)
save_image(images, "originals.png")

# Turn into tokens
patch_size = 16
tokens = images_to_tokens(images, patch_size)
b_size, num_patches, _ = tokens.shape

# Apply the masking (requires looking at them as numpy arrays)
block_mask = [beit_block_masking(img.numpy(), patch_size=patch_size) for img in images]
block_mask = T.from_numpy(np.array(block_mask))
tokens[block_mask] = 0

# Turn back into images
masked_images = tokens_to_images(tokens, images.shape[2:], patch_size)

save_image(masked_images, "block_masked.png")
