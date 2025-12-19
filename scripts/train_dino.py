import math
import argparse
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import v2
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightly.transforms.dino_transform import DINOTransform

from src.data import MultiSet, ResizeAndPad
from src.model_dino import DINO


class ImageTransformTrain:
    """Image transformation pipeline for DINO training."""

    def __init__(self, target_size: int) -> None:
        self.transforms = v2.Compose(
            [
                v2.Lambda(lambda x: x.crop((0, 25, x.width, x.height))),
                ResizeAndPad(size=math.ceil(1.05 * target_size)),
                v2.Grayscale(num_output_channels=1),
                DINOTransform(local_crop_scale=(0.1, 0.4), cj_sat=0, cj_hue=0, normalize={"mean": [0.5], "std": [0.5]}),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", nargs="+", help="One or more dataset directories.", required=True)
    parser.add_argument("-ckpt", "--ckpt_path", default=None, help="ckpt_path")
    parser.add_argument("-c", "--config", help="Path to model card (yaml file).")
    return parser.parse_args()


def load_config(card_path: Path) -> dict:
    with open(card_path, "r") as stream:
        return yaml.safe_load(stream)


def create_datasets(dataset_paths: list, target_size):
    
    # Create transforms
    image_transforms_train = ImageTransformTrain(target_size)

    train_sets = []
    dataset_names = []

    for dataset_path in dataset_paths:
        data_path = Path(dataset_path)
        train_sets.append(MultiSet(annotation_path=data_path / "train.csv", 
                                   image_transforms=image_transforms_train))
        dataset_names.append("_".join(data_path.parts[-2:]))

    return ConcatDataset(train_sets), dataset_names


def create_collate_fn():
    """Create collate function for DataLoader."""

    def multi_collate(batch):
        image_views, _, _, image_shape, _ = zip(*(sample.values() for sample in batch))
        num_views = len(image_views[0])
        batched_views = [torch.stack([views[i] for views in image_views]) for i in range(num_views)]
        return {
            "image": batched_views,
            "image_shape": torch.stack(image_shape),
        }

    return multi_collate


def main():
    # Parse arguments
    args = parse_arguments()

    # Load model card
    card_path = Path(args.config)
    card_dict = load_config(card_path)

    # Set precision
    precision = card_dict.get("precision", "highest")
    torch.set_float32_matmul_precision(precision)

    # Extract configuration
    target_size = card_dict["target_size"]
    bs = card_dict["bs"]

    # Create datasets
    train_set, dataset_names = create_datasets(args.dataset, target_size)

    # Create model
    model = DINO(
        image_encoder_args=card_dict["image_encoder_args"],
        head_args=card_dict["head_args"],
        optim_args=card_dict["optim_args"],
    )

    # Create dataloader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=True,
        num_workers=card_dict["num_workers"],
        drop_last=True,
        collate_fn=create_collate_fn(),
    )

    # Setup logger
    experiment_name = f"{card_path.stem}_{'+'.join(dataset_names)}"
    logger = TensorBoardLogger(save_dir="./logs/dino/", name=experiment_name)

    # Setup callbacks
    checkpoint = ModelCheckpoint(
        filename="{epoch}_{train_loss:.5f}",
        monitor="train_loss",
        save_top_k=card_dict.get("save_top_k", 1),
        mode="min",
        save_last=True,
    )

    stopper = EarlyStopping(
        monitor="train_loss", 
        min_delta=0.0, 
        patience=card_dict["patience"], 
        check_finite=False, 
        mode="min"
    )

    # Create trainer
    trainer = Trainer(
        log_every_n_steps=len(train_loader), 
        logger=logger, 
        callbacks=[checkpoint, stopper], 
        **card_dict["trainer_args"]
    )

    # Train
    print(f"Training from model card: {args.config}")
    print(f"Datasets: {', '.join(dataset_names)}")
    trainer.fit(model, train_loader, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
