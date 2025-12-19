import torch
from torch import Tensor
from torch.nn import Module, Parameter, functional as F

class CLIPLoss(Module):
    """From https://arxiv.org/pdf/2103.00020"""

    def __init__(self, bias: bool = False) -> None:
        super().__init__()
        self.logit_scale = Parameter(torch.ones([]))

    def forward(self, image_emb: Tensor, profile_emb: Tensor, buckets: int = 1) -> Tensor:
        assert image_emb.size(0) % buckets == 0, "Batch size must be divisible by number of buckets!"
        bucket_size = image_emb.size(0) // buckets
        
        # In some cases the loss turned to NaN so try to remove that
        if torch.isnan(image_emb).any() or torch.isnan(profile_emb).any():
            raise ValueError("NaNs in embeddings before normalization!")

        image_emb = F.normalize(image_emb, dim=-1, eps=1e-8)
        profile_emb = F.normalize(profile_emb, dim=-1, eps=1e-8)

        image_emb = image_emb.view(buckets, bucket_size, -1)
        profile_emb = profile_emb.view(buckets, bucket_size, -1)

        logit_scale = self.logit_scale.exp().clamp(min=1 / 100, max=100.0)

        logits = (image_emb @ profile_emb.transpose(1, 2)) * logit_scale

        logits = logits - logits.max(dim=-1, keepdim=True).values

        label = torch.arange(bucket_size).long().to(image_emb.device)
        label = torch.stack([label] * buckets)

        loss_1 = torch.stack([F.cross_entropy(x, y) for x, y in zip(logits, label)]).mean()
        loss_2 = torch.stack([F.cross_entropy(x.T, y) for x, y in zip(logits, label)]).mean()
        loss = (loss_1 + loss_2) / 2

        return loss


class SigLIPLoss(Module):
    """From https://arxiv.org/pdf/2303.15343"""

    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = Parameter(torch.ones([]))
        self.bias = Parameter(-10 * torch.ones([]))

    def forward(self, image_emb: Tensor, profile_emb: Tensor, buckets: int = 1) -> Tensor:
        assert image_emb.size(0) % buckets == 0, "Batch size must be divisible by number of buckets!"

        image_emb = F.normalize(image_emb)
        profile_emb = F.normalize(profile_emb)

        bucket_size = image_emb.size(0) // buckets
        image_emb = image_emb.view(buckets, bucket_size, -1)
        profile_emb = profile_emb.view(buckets, bucket_size, -1)

        logits = image_emb @ profile_emb.transpose(1, 2)

        # Scale and add bias
        logits = logits * self.logit_scale.exp() + self.bias

        device = logits.device
        labels = torch.eye(bucket_size, device=device)
        labels = 2 * labels - 1

        loss = -F.logsigmoid(labels * logits).sum() / bucket_size

        return loss.mean()
