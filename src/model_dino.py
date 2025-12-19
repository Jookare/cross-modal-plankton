import copy
import torch
import timm

from torch import nn, Tensor
from torch.nn import Module
from typing import Dict
from lightning.pytorch import LightningModule
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from typing import Any


def get_bottleneck(backbone: Module, bottleneck_in: int, bottleneck_size: int) -> Module:
    if bottleneck_size <= 0:
        return nn.Identity()

    bottleneck_out = bottleneck_size

    # Heuristic: CNNs have BatchNorm2d somewhere
    is_cnn = any(isinstance(m, nn.BatchNorm2d) for m in backbone.modules())

    if is_cnn:
        norm_layer = nn.BatchNorm1d
        act_layer = nn.ReLU
    else:
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU

    return nn.Sequential(
        nn.Linear(bottleneck_in, bottleneck_out),
        norm_layer(bottleneck_out),
        act_layer(),
    )


class ImageEncoder(Module):
    def __init__(
        self,
        name: str,
        num_classes: int = 0,
        pretrained: bool = True,
        dropout: float = 0.0,
        in_chans: int = 1,
        drop_path_rate: float = 0.1,
        metadata: bool = True,
        dynamic_img_size: bool = False,
        bottleneck_dim: int = 0,
    ) -> None:
        super().__init__()

        # Base kwargs shared by all models
        kwargs = dict(
            num_classes=num_classes,
            pretrained=pretrained,
            in_chans=in_chans,
            drop_path_rate=drop_path_rate,
        )
        
        # Add only supported kwargs for ViT-style models
        if "vit" in name.lower() or "deit" in name.lower() or "swin" in name.lower():
            kwargs["dynamic_img_size"] = dynamic_img_size

        # Create model
        self.backbone = timm.create_model(name, **kwargs)

        # Get output dimension
        self.dim_out = self.backbone.num_features + (2 if metadata else 0)
        self.metadata = metadata
        self.drop = nn.Dropout(dropout)  # Not used
        self.bottleneck = get_bottleneck(self.backbone, self.dim_out, bottleneck_dim)

    def forward(self, image: Tensor, **kwargs) -> Tensor:
        x = self.backbone.forward_features(image)

        # Select cls token or average pool
        if x.ndim == 3:  # ViT: (B, N, D)
            x = x[:, 0]
        elif x.ndim == 4:  # CNN: (B, C, H, W)
            x = x.mean(dim=[2, 3])

        x = self.bottleneck(x)
        return x


class DINO(LightningModule):
    def __init__(
        self,
        image_encoder_args: Dict[str, Any],
        head_args: Dict[str, Any],
        optim_args: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        backbone = ImageEncoder(**image_encoder_args)

        input_dim = 512
        self.student_backbone = backbone

        hidden_dim = head_args.get("hidden_dim")
        bottleneck_dim = head_args.get("bottleneck_dim")
        output_dim = head_args.get("output_dim")

        self.student_head = DINOProjectionHead(
            input_dim=input_dim,
            freeze_last_layer=1,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=output_dim,
        )

        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=output_dim,
        )

        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=output_dim, teacher_temp=0.07)

        self.optim_args = optim_args
        self.train_loss = []

    def forward(self, image, **kwargs):
        x = self.student_backbone(image, **kwargs)
        out = self.student_head(x)
        return out

    def forward_teacher(self, image, **kwargs):
        x = self.teacher_backbone(image, **kwargs)
        out = self.teacher_head(x)
        return out

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, self.trainer.max_epochs, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        # Update weight decay following DINO's cosine schedule
        self._update_weight_decay()

        # Extract views from dictionary
        views = batch["image"]  # This is already a list of tensors
        views = [view.to(self.device) for view in views]

        global_views = views[:2]

        # Pass image_shape to forward methods if using metadata
        image_shape = batch["image_shape"].to(self.device)

        teacher_out = [self.forward_teacher(view, image_shape=image_shape) for view in global_views]
        student_out = [self.forward(view) for view in views]

        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)

        self.train_loss.append(loss.detach())
        return loss

    def _update_weight_decay(self):
        """Update weight decay following cosine schedule."""
        max_epochs = self.trainer.max_epochs
        weight_decay = self.optim_args.get("weight_decay", 0.04)
        weight_decay_end = self.optim_args.get("weight_decay_end", 0.4)

        # Cosine schedule for weight decay
        wd = cosine_schedule(self.current_epoch, max_epochs, weight_decay, weight_decay_end)

        # Update weight decay in optimizer
        for param_group in self.optimizers().param_groups:
            param_group["weight_decay"] = wd

    def on_train_epoch_end(self) -> None:
        loss = torch.stack(self.train_loss)
        loss = loss.mean()

        metrics = {"train_loss": loss, "step": self.current_epoch}
        self.log_dict(metrics)

        self.train_loss.clear()

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def predict_step(self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Any:
        image = batch["image"]
        embeddings = self.teacher_backbone(image)
        if "label" in batch:
            return {"embs": embeddings} | {"label": batch["label"]}
        else:
            return {"embs": embeddings}

    def configure_optimizers(self):
        args = dict(self.optim_args)
        optim_type = args.pop("optim")
        warmup_epochs = args.pop("warmup_epochs", None)
        max_epochs = self.trainer.max_epochs
        args.pop("weight_decay_end")

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
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

        else:
            return optimizer
