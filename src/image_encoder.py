import torch
from torch import nn, Tensor
from torch.nn import Module
import timm

class ImageEncoder(Module):
    def __init__(
        self,
        name: str,
        num_classes: int = 0,
        pretrained: bool = False,
        dropout: float = 0.0,
        in_chans: int = 1,
        drop_path_rate: float = 0.0,
        metadata: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            name, num_classes=num_classes, pretrained=pretrained, in_chans=in_chans, drop_path_rate=drop_path_rate
        )
        self.dim_out = self.backbone.num_features + 2 * metadata
        self.metadata = metadata
        self.drop = nn.Dropout(dropout)

    def forward(self, image: Tensor, **kwargs) -> Tensor:  # Changed return type
        x = self.backbone(image)
        if self.metadata:
            # Safely get image_shape, only process if it exists
            image_shape = kwargs.get("image_shape")
            if image_shape is not None:
                metadata = image_shape.to(image.dtype)
                metadata /= image.shape[2]
                x = torch.cat((x, metadata), 1)
        return self.drop(x)
