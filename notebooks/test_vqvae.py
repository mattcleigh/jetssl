from pathlib import Path

import torch as T
import torch.nn.functional as F
import torchvision.transforms as TV
from dall_e import map_pixels, unmap_pixels
from PIL import Image
from torch import nn
from torchvision.utils import save_image

target_image_size = 256


def fix_old_upsample(model: nn.Module) -> bool:
    """Check if a network has at least one BayesianLinear layer.

    Loops over a network's submodules and looks for BayesianLinear layers.
    """
    if isinstance(model, nn.Upsample):
        model.recompute_scale_factor = None
    for m in model.children():
        fix_old_upsample(m)


def apply_dalle_preprocessing(image: T.Tensor) -> T.Tensor:
    return map_pixels(F.interpolate(image, size=(112, 112)))  # 112 gives 14x14 tokens


def apply_dalle_postprocessing(image: T.Tensor) -> T.Tensor:
    return F.interpolate(unmap_pixels(T.sigmoid(image)), size=(224, 224))


device = "gpu" if T.cuda.is_available() else "cpu"
encoder = T.load(
    "/srv/beegfs/scratch/groups/rodem/openai/encoder.pkl", map_location=device
)
decoder = T.load(
    "/srv/beegfs/scratch/groups/rodem/openai/decoder.pkl", map_location=device
)
fix_old_upsample(decoder)

# Load an image and convert to a tensor
transforms = TV.Compose([
    TV.Resize(224),
    TV.CenterCrop(224),
    TV.ToTensor(),
])
image_folder = "/srv/beegfs/scratch/groups/rodem/datasets/ImageNet/val/n04118776/"
image_name = "ILSVRC2012_val_00011948.JPEG"
image = Image.open(Path(image_folder, image_name))
image = transforms(image).unsqueeze(0)

save_image(image, "original.png")

# Get the encodings
image = apply_dalle_preprocessing(image)
patch_encodings = T.argmax(encoder(image), axis=1)

# Decode the outputs
encoded_vecs = (
    F.one_hot(patch_encodings, num_classes=encoder.vocab_size)
    .permute(0, 3, 1, 2)
    .float()
)
reconstructed = decoder(encoded_vecs)[:, :3]
rec_image = apply_dalle_postprocessing(reconstructed)

save_image(rec_image, "reconstructed.png")
