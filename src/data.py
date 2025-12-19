import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable, Dict, Optional, Callable
from PIL import Image

class MultiSet(Dataset):
    def __init__(
        self,
        annotation_path: Path,
        image_transforms: Optional[Callable] = None,
        profile_transform: Optional[Callable] = None,
        pair_augmentation: Optional[Callable] = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.parent = annotation_path.parent
        self.table = pd.read_csv(annotation_path)
        self.class_names = np.unique(self.table["class"])

        # Shuffle as csv files are sorted
        if shuffle:
            self.table = self.table.sample(frac=1, random_state=seed).reset_index(drop=True)

        self.image_transforms = image_transforms
        self.profile_transform = profile_transform
        self.pair_augmentation = pair_augmentation

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        # image = cv2.imread(self.parent / self.table.image[index], cv2.IMREAD_GRAYSCALE)
        image = Image.open(self.parent / self.table.image[index]).convert("RGB")
        profile = np.loadtxt(self.parent / self.table.profile[index], delimiter=",", skiprows=1)

        # image_shape = torch.tensor(image.shape)
        image_shape = torch.tensor(image.size[::-1])
        profile_length = torch.tensor([profile.shape[0]])

        if self.image_transforms:
            image = self.image_transforms(image)
        else:
            # If no transforms, just convert to Tensor
            image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(image)

        if self.profile_transform:
            profile = self.profile_transform(profile)

        label = self.table["class"][index]

        if self.pair_augmentation:
            image, profile = self.pair_augmentation(image, profile)

        return {
            "image": image,
            "profile": profile,
            "label": label,
            "image_shape": image_shape,
            "profile_length": profile_length,
        }


class ImageTransformTrain:
    def __init__(self, target_size=224) -> None:
        self.transforms = v2.Compose(
            (
                v2.Lambda(lambda x: x.crop((0, 25, x.width, x.height))),
                ResizeAndPad(size=math.ceil(1.05 * target_size)),
                v2.Grayscale(num_output_channels=1),
                v2.RandomResizedCrop((target_size, target_size), scale=(0.8, 1.0)),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
                v2.RandomVerticalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Lambda(lambda x: x * 2.0 - 1.0),
            )
        )

    def __call__(self, img: Image) -> Tensor:
        return self.transforms(img)


class ImageTransformTest:
    def __init__(self, target_size=224) -> None:
        self.transforms = v2.Compose(
            (
                v2.Lambda(lambda x: x.crop((0, 25, x.width, x.height))),
                ResizeAndPad(size=target_size),
                v2.Grayscale(num_output_channels=1),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Lambda(lambda x: x * 2.0 - 1.0),
            )
        )

    def __call__(self, img: Image) -> Tensor:
        return self.transforms(img)


class ProfileTransformTrain:
    def __init__(self, target_size=224) -> None:
        self.div = torch.FloatTensor([8.9211, 8.9211, 8.9211, 8.9211, 8.9211, 8.9211])
        self.transform = v2.Compose(
            (
                v2.Lambda(lambda x: torch.tensor(x).add(1).log().div(self.div).mul(2).add(-1)),
                RandomAmplitudeScale(),
                RandomBandDrop(p=0.25),
                v2.Lambda(lambda x: x.t().unsqueeze(1)),
                v2.Resize((1, math.ceil(1.05 * target_size))),
                v2.RandomCrop((1, target_size)),
                v2.Lambda(lambda x: x.squeeze(1).t().float()),
            )
        )

    def __call__(self, prof: np.ndarray):
        return self.transform(prof)


class ProfileTransformTest:
    def __init__(self, target_size=224) -> None:
        self.div = torch.FloatTensor([8.9211, 8.9211, 8.9211, 8.9211, 8.9211, 8.9211])
        self.transform = v2.Compose(
            (
                v2.Lambda(lambda x: torch.tensor(x).add(1).log().div(self.div).mul(2).add(-1)),
                v2.Lambda(lambda x: x.t().unsqueeze(1)),
                v2.Resize((1, target_size)),
                v2.Lambda(lambda x: x.squeeze(1).t().float()),
            )
        )

    def __call__(self, prof: np.ndarray):
        return self.transform(prof)


class PairAugmentation:
    def __call__(self, image: Tensor, profile: Tensor) -> Iterable[Tensor]:
        if torch.rand(1).item() >= 0.5:
            image = v2.functional.horizontal_flip(image)
            profile = profile.flip(0)
        return image, profile


def resize_profile(profile: Tensor, target_len: int = 256) -> Tensor:
    _, d = profile.shape
    profile = profile.unsqueeze(0)
    profile = v2.functional.resize(profile, (target_len, d))
    return profile.squeeze(0)


class ResizeAndPad:
    def __init__(self, size=256, fill=None):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        # Resize while keeping aspect ratio
        orig_w, orig_h = img.size
        ratio = self.size / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
        img = v2.functional.resize(img, (new_h, new_w))

        # Pad to (size, size)
        pad_left = (self.size - new_w) // 2
        pad_top = (self.size - new_h) // 2
        pad_right = self.size - new_w - pad_left
        pad_bottom = self.size - new_h - pad_top

        if self.fill is None:
            img = v2.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), padding_mode="edge")
        else:
            img = v2.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

        return img


class RandomBandDrop:
    def __init__(self, p: float = 0.2, per_band_q: float = 0.05, max_drop: int = 4):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p = p
        self.per_band_q = per_band_q
        self.max_drop = max_drop

    def _num_to_drop(self, n_chan: int) -> int:
        probs = torch.full((n_chan,), self.per_band_q)
        n_drop = int(torch.bernoulli(probs).sum().item())
        return min(n_drop, self.max_drop)  # cap at max_drop

    def __call__(self, profile):
        # Decide whether to drop a band for this sample
        if torch.rand(1).item() >= self.p:
            return profile
        n_chan = profile.shape[1]
        n_drop = self._num_to_drop(n_chan)

        # drop at least one band
        if n_drop < 1:
            n_drop = 1

        drop_idxs = torch.randperm(n_chan)[:n_drop]
        profile[:, drop_idxs] = 0.0

        return profile


class RandomAmplitudeScale:
    def __init__(self, low=0.95, high=1.05, per_band=True, p=0.5):
        self.low, self.high, self.per_band, self.p = low, high, per_band, p

    def __call__(self, x: torch.Tensor):
        # No scaling
        if torch.rand(1) > self.p:
            return x

        # If scale per band is True then scale each band individually
        if self.per_band:
            factors = torch.empty(x.shape[-1], device=x.device).uniform_(self.low, self.high)
            return x * factors
        else:
            factor = torch.empty(1, device=x.device).uniform_(self.low, self.high)
            return x * factor