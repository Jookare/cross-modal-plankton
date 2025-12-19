import torch
from torch import nn, Tensor
from typing import Dict, Any, Callable
from .image_encoder import ImageEncoder
from .profile_encoder import ProfileCNN, ProfileTransformer, ProfileLSTM
from .loss import CLIPLoss, SigLIPLoss
from lightning import LightningModule

class MultiModel(LightningModule):
    def __init__(
        self,
        dim_embed,
        image_encoder_args: Dict[str, Any],
        profile_encoder_args: Dict[str, Any],
        coordination_args: Dict[str, Any],
        optim_args: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Construct encoders
        self.image_encoder = ImageEncoder(**image_encoder_args)

        self.image_projection = nn.Linear(self.image_encoder.dim_out, dim_embed, bias=False)

        if "num_head" in profile_encoder_args:
            self.profile_encoder = ProfileTransformer(**profile_encoder_args)
        elif "blocks" in profile_encoder_args:
            self.profile_encoder = ProfileCNN(**profile_encoder_args)
        else:
            self.profile_encoder = ProfileLSTM(**profile_encoder_args)
        self.profile_projection = nn.Linear(self.profile_encoder.dim_out, dim_embed, bias=False)

        # Loss
        method = coordination_args.get("method")
        if method == "clip":
            self.loss = CLIPLoss()
        elif method == "siglip":
            self.loss = SigLIPLoss()
        else:
            raise Exception(f"Loss function {method} not implemented.")

        # Miscellaneous
        self.optim_args = optim_args
        self.train_loss = []
        self.valid_loss = []

    def safe_forward(self, model: Callable, **kwargs):
        return model(**kwargs) if None not in kwargs.values() else None

    def tokenize(self, profile: Tensor) -> Dict[str, Tensor]:
        return self.profile_encoder.tokenize(profile)

    def encode(self, image: Tensor | None, profile: Tensor | None, **kwargs) -> Dict[str, Tensor | None]:
        image_emb = self.safe_forward(self.image_encoder, image=image, **kwargs)
        profile_emb = self.safe_forward(self.profile_encoder, profile=profile, **kwargs)

        image_emb = self.safe_forward(self.image_projection, input=image_emb)
        profile_emb = self.safe_forward(self.profile_projection, input=profile_emb)

        return {"image_emb": image_emb, "profile_emb": profile_emb}

    def forward(self, **kwargs) -> Dict[str, Tensor | None]:
        embeddings = self.encode(**kwargs)
        return embeddings

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        embeddings = self.encode(**batch)
        embeddings["buckets"] = batch["buckets"]

        loss = self.loss(**embeddings)
        self.train_loss.append(loss.detach())

        return loss

    def on_train_epoch_end(self) -> None:
        loss = torch.stack(self.train_loss)
        loss = loss.mean()

        metrics = {"train_loss": loss, "step": self.current_epoch}
        self.log_dict(metrics)

        self.train_loss.clear()

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        embeddings = self.encode(**batch)
        embeddings["buckets"] = batch["buckets"]

        loss = self.loss(**embeddings)
        self.valid_loss.append(loss.detach())

    def on_validation_epoch_end(self) -> None:
        loss = torch.stack(self.valid_loss)
        loss = loss.mean()

        metrics = {"valid_loss": loss, "step": self.current_epoch}
        self.log_dict(metrics)

        self.valid_loss.clear()

    def predict_step(self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Any:
        embeddings = self.encode(**batch)
        if "label" in batch:
            return embeddings | {"label": batch["label"]}
        else:
            return embeddings

    def configure_optimizers(self):
        args = dict(self.optim_args)
        optim_type = args.pop("optim")
        warmup_epochs = args.pop("warmup_epochs", None)
        max_epochs = args.pop("max_epochs", None)

        if optim_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), **args)
        elif optim_type.lower() == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), **args)
        else:
            raise ValueError(f"Unsupported optimizer: {optim_type}")

        if warmup_epochs is not None:
            # Linear warmup up
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
            )

            # Cosine decay
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-5
            )

            # Chain schedulers
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
            )

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

        else:
            return optimizer
