import yaml
import pickle
from pathlib import Path
import torch


from lightning import Trainer
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.data import MultiSet, ImageTransformTest, ProfileTransformTest
from src.model import MultiModel
import numpy as np
from tqdm import tqdm

import logging
import os

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
os.environ["SLURM_NTASKS"] = "1"


def collate(batch, model):
    image, profile, label, image_shape, profile_len = zip(*(sample.values() for sample in batch))

    image = {"image": torch.stack(image)}
    profile = model.profile_encoder.tokenize(profile)
    label = {"label": label}
    image_shape = {"image_shape": torch.stack(image_shape)}
    profile_len = {"profile_len": torch.stack(profile_len)}

    return image | profile | label | image_shape | profile_len


def get_embeddings(model, loader, trainer):
    outputs = trainer.predict(model, loader)
    image_emb, profile_emb, labels = zip(*(sample.values() for sample in outputs))

    return (
        F.normalize(torch.cat(image_emb)).cpu().numpy(),
        F.normalize(torch.cat(profile_emb)).cpu().numpy(),
        np.concatenate(labels),
    )

def make_loader(annotation_path, image_size, batch_size, num_workers, model):
    dataset = MultiSet(
        annotation_path=annotation_path,
        image_transforms=ImageTransformTest(image_size),
        profile_transform=ProfileTransformTest(image_size),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda x: collate(x, model),
    )
    
def extract_embeddings(
    checkpoint_path: Path,
    data_path: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    mode: str,  # "split" or "single"
):  
    
    model = MultiModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    trainer = Trainer(barebones=True, devices=[0])
        
    if mode == "split":
        # --- loaders ---
        train_loader = make_loader(data_path / "train.csv", image_size, batch_size, num_workers, model)
        test_loader = make_loader(data_path / "test.csv", image_size, batch_size, num_workers, model)
    
        # --- embeddings ---
        I_train, P_train, L_train = get_embeddings(model, train_loader, trainer)
        I_test, P_test, L_test = get_embeddings(model, test_loader, trainer)

        return {
            "train": {
                "image": I_train, 
                "profile": P_train, 
                "label": L_train
            },
            "test": {
                "image": I_test, 
                "profile": P_test, 
                "label": L_test
            },
            "classes": np.unique(L_train),
        }

    if mode == "single":
        loader = make_loader(data_path / "annotations.csv", image_size, batch_size, num_workers, model)
        I_embs, P_embs, Labels = get_embeddings(model, loader, trainer)

        return {
            "image": I_embs, 
            "profile": P_embs, 
            "label": Labels,
            "classes": np.unique(Labels),
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main():
    with open("configs/experiment/experiment.yaml") as f:
        cfg = yaml.safe_load(f)

    folder = cfg["models"]["folder"]
    checkpoints = cfg["models"]["checkpoints"]
    folds = cfg["models"]["folds"]

    train_data = cfg["data"]["train"]
    test_data = cfg["data"]["test"]

    mode = "split" if test_data == "LAB" else "single"

    embeddings = {ckpt: {} for ckpt in checkpoints}
    

    for ckpt_name in tqdm(checkpoints):
        for i in range(1, folds + 1):
            if train_data == "LAB":
                checkpoint_path = Path(f"logs/{folder}/{ckpt_name}_fold{i}/version_0/checkpoints")
            elif train_data == "UTO":
                checkpoint_path = Path(f"logs/{folder}/{ckpt_name}_fold/version_0/checkpoints")
            else:
                raise ValueError(f"Unknown train data {train_data}")

            # LAB has folds but SEA does not
            if test_data == "LAB":
                data_path = Path(f"data/{test_data}/fold{i}/")
            else:
                data_path = Path(f"data/{test_data}/")

            checkpoint = sorted(list(checkpoint_path.iterdir()))[0]
            print(checkpoint)

            emb = extract_embeddings(
                checkpoint_path=checkpoint,
                data_path=data_path,
                image_size=cfg["data"]["image_size"],
                batch_size=cfg["embedding"]["batch_size"],
                num_workers=cfg["embedding"]["num_workers"],
                mode=mode
            )
            embeddings[ckpt_name][i] = emb

            if train_data == "UTO":
                break


    print(f"saving embeddings to {cfg["embedding"]["save_path"]}")
    with open(cfg["embedding"]["save_path"], "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
