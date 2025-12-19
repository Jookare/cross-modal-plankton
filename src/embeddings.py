from pathlib import Path
import torch
from lightning import Trainer
from torch.utils.data import DataLoader
from torch.nn.functional import normalize

from src.data import MultiSet, ImageTransformTest, ProfileTransformTest
from src.model import MultiModel


def extract_embeddings(
    checkpoint_path: Path,
    annotation_path: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
):
    model = MultiModel.load_from_checkpoint(checkpoint_path)
    model.eval()

    image_tf = ImageTransformTest(image_size)
    profile_tf = ProfileTransformTest(image_size)

    dataset = MultiSet(
        annotation_path=annotation_path,
        image_transforms=image_tf,
        profile_transform=profile_tf,
    )

    def collate(batch):
        image, profile, label, image_shape, profile_len = zip(*(sample.values() for sample in batch))
        return {
            "image": torch.stack(image),
            **model.profile_encoder.tokenize(profile),
            "label": label,
            "image_shape": torch.stack(image_shape),
            "profile_len": torch.stack(profile_len),
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
    )

    trainer = Trainer(barebones=True)
    outputs = trainer.predict(model, loader)

    images, profiles, labels = zip(*(o.values() for o in outputs))

    return {
        "image": normalize(torch.cat(images)).cpu().numpy(),
        "profile": normalize(torch.cat(profiles)).cpu().numpy(),
        "label": torch.cat(labels).cpu().numpy(),
    }
