import torch
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
import yaml
import argparse

from src.data import (
    MultiSet,
    ImageTransformTrain,
    ImageTransformTest,
    ProfileTransformTrain,
    ProfileTransformTest,
    PairAugmentation,
)
from src.model import MultiModel

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", nargs="+", help="One or more dataset directories.", required=True)
    parser.add_argument("-ckpt", "--ckpt_path", default=None, help="ckpt_path")
    parser.add_argument("-c", "--config", help="Path to model card (yaml file).")
    return parser.parse_args()


def load_config(card_path: Path) -> dict:
    with open(card_path, "r") as stream:
        return yaml.safe_load(stream)


def multi_collate(batch, model, card_dict):
    image, profile, _, image_shape, profile_len = zip(*(sample.values() for sample in batch))

    image = {"image": torch.stack(image)}
    profile = model.profile_encoder.tokenize(profile)
    image_shape = {"image_shape": torch.stack(image_shape)}
    profile_len = {"profile_len": torch.stack(profile_len)}
    buckets = {"buckets": card_dict["buckets"]}

    return image | profile | image_shape | profile_len | buckets

def create_datasets(dataset_paths: list, target_size):
    
    # Create transforms
    image_transforms_train = ImageTransformTrain(target_size)
    signal_transforms_train = ProfileTransformTrain(target_size)
    pair_augmentation = PairAugmentation()

    image_transforms_test = ImageTransformTest(target_size)
    signal_transforms_test = ProfileTransformTest(target_size)
    
    train_sets = []
    val_sets = []
    dataset_names = []

    # Create datasets
    # Multiple dataset can be given so loop through and concat at the end
    for dataset_path in dataset_paths:
        data_path = Path(dataset_path)

        train_sets.append(
            MultiSet(
                annotation_path=data_path / "train.csv",
                image_transforms=image_transforms_train,
                profile_transform=signal_transforms_train,
                pair_augmentation=pair_augmentation,
            )
        )

        val_sets.append(
            MultiSet(
                annotation_path=data_path / "val.csv",
                image_transforms=image_transforms_test,
                profile_transform=signal_transforms_test,
            )
        )

        dataset_names.append("_".join(data_path.parts[-2:]))

    train_set = ConcatDataset(train_sets)
    val_set = ConcatDataset(val_sets)

    return train_set, val_set, dataset_names


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
    valid_bs = card_dict["valid_bs"]

    # Create datasets
    train_set, val_set, dataset_names = create_datasets(args.dataset, target_size)

    # Init models
    model = MultiModel(
        dim_embed=card_dict["dim_embedding"],
        image_encoder_args=card_dict["image_encoder_args"],
        profile_encoder_args=card_dict["profile_encoder_args"],
        coordination_args=card_dict["coordination_args"],
        optim_args=card_dict["optim_args"],
    )

    # Create dataloaders
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=True,
        num_workers=card_dict["num_workers"],
        drop_last=True,
        collate_fn=lambda x: multi_collate(x, model, card_dict),
    )

    valid_loader = DataLoader(
        dataset=val_set,
        batch_size=valid_bs,
        shuffle=False,
        num_workers=card_dict["num_workers"],
        drop_last=False,
        collate_fn=lambda x: multi_collate(x, model, card_dict),
    )

    name = f"{card_path.stem}_{'+'.join(dataset_names)}"
    logger = TensorBoardLogger(save_dir="./logs/multi/", name=name)

    checkpoint = ModelCheckpoint(
        filename="{epoch}_{valid_loss:.5f}",
        monitor="valid_loss",
        save_top_k=card_dict.get("save_top_k", 1),
        mode="min",
        save_last=True,
    )

    stopper = EarlyStopping(
        monitor="valid_loss", 
        min_delta=0.0, 
        patience=card_dict["patience"], 
        check_finite=False, 
        mode="min"
    )

    trainer = Trainer(
        log_every_n_steps=len(train_loader), 
        logger=logger, 
        callbacks=[checkpoint, stopper], 
        **card_dict["trainer_args"]
    )

    print(f"Training from model card {args.config}")
    trainer.fit(model, train_loader, valid_loader, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
