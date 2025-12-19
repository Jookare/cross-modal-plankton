import torch
from torch import Tensor
from torch import nn
from torch.nn import Module
from typing import Iterable, Dict

class ProfileTransformer(Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        num_head: int,
        max_seq_len: int = 2048,
        num_layers: int = 6,
        dim_feedforward: int = 2024,
        dropout: float = 0.1,
        activation: str = "gelu",
        metadata: bool = True,
    ) -> None:
        super().__init__()

        self.expand = nn.Linear(dim_in, dim_hidden, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_hidden))

        # Use padding_idx=0 and start positions at 1
        self.position = nn.Embedding(max_seq_len, dim_hidden, padding_idx=0)
        self.padding_idx = self.position.padding_idx
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_hidden,
                nhead=num_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.drop = nn.Dropout(dropout)
        self.dim_out = dim_hidden + metadata
        self.metadata = metadata

    def tokenize(self, profile: Tensor | Iterable[Tensor]) -> Dict[str, Tensor]:
        if not isinstance(profile, (list, tuple)):
            profile = [profile]

        device = profile[0].device

        # Positions: 1 for CLS, 2..L+1 for tokens, 0 for padding
        time = [torch.arange(1, 2 + p.shape[0], dtype=torch.long, device=device) for p in profile]
        time = nn.utils.rnn.pad_sequence(time, batch_first=True, padding_value=self.padding_idx)

        profile = nn.utils.rnn.pad_sequence(profile, batch_first=True, padding_value=self.padding_idx)
        padding_mask = time == self.padding_idx

        return {"profile": profile, "time": time, "padding_mask": padding_mask}

    def forward(self, profile: Tensor, time: Tensor, padding_mask: Tensor, **kwargs) -> Tensor:
        batch_size = profile.shape[0]

        # Expand profile and add CLS token
        x = self.expand(profile)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        x = x + self.position(time)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = x[:, 0]

        if self.metadata:
            metadata = kwargs["profile_len"].to(profile.dtype)
            metadata /= profile.shape[1]
            x = torch.cat((x, metadata), 1)
        return self.drop(x)


class ProfileLSTM(Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        num_layers: int,
        dropout: float = 0.1,
        metadata: bool = True,
    ) -> None:
        super().__init__()

        self.expand = nn.Linear(dim_in, dim_hidden, bias=False)
        self.lstm = nn.LSTM(dim_hidden, dim_hidden, num_layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.dim_out = 2 * dim_hidden + metadata
        self.metadata = metadata

    def tokenize(self, profile: Tensor | Iterable[Tensor]) -> Dict[str, Tensor]:
        if not isinstance(profile, (list, tuple)):
            profile = [profile]

        last = torch.tensor([p.shape[0] - 1 for p in profile]).long()
        profile = nn.utils.rnn.pad_sequence(profile, batch_first=True)

        return {"profile": profile, "last_idx": last}

    def forward(self, profile: Tensor, last_idx: Tensor, **kwargs) -> Tensor:
        x = self.expand(profile)
        x, _ = self.lstm(x)

        # Works also with bidirectional
        batch_idx = torch.arange(x.shape[0])
        fwd_last = x[batch_idx, last_idx, : self.lstm.hidden_size]
        bwd_last = x[batch_idx, 0, self.lstm.hidden_size :]
        x = torch.cat([fwd_last, bwd_last], dim=1)

        if self.metadata:
            metadata = kwargs["profile_len"].to(profile.dtype)
            metadata /= profile.shape[1]
            x = torch.cat((x, metadata), 1)

        return self.drop(x)


class _BasicBlock(Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        downsample: Module | None = None,
        groups: int = 1,
        base_channels: int = 64,
    ) -> None:
        super().__init__()

        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.base_channels = base_channels

        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu(out)

        return out


class ProfileCNN(Module):
    """
    ResNet-based CNN, adapted from https://github.com/Lornatang/ResNet-PyTorch/blob/main/model.py
    """

    def __init__(
        self,
        dim_in,
        blocks: list[int],
        groups: int = 1,
        block_type: type[_BasicBlock] = _BasicBlock,
        base_channels: int = 32,
        dropout=0.1,
        metadata: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = self.base_channels = base_channels
        self.dilation = 1
        self.groups = groups

        self.conv1 = nn.Conv1d(dim_in, self.in_channels, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool1d(3, 2, 1)

        # Dynamically build stages
        layers = []
        channels = base_channels
        for i, num_blocks in enumerate(blocks):
            # first stage no downsample
            stride = 1 if i == 0 else 2
            stage = self._make_layer(num_blocks, channels, block_type, stride)
            layers.append(stage)
            channels *= 2  # double channels after each stage
        self.stages = nn.ModuleList(layers)

        # Head
        self.avgpool = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(dropout)

        self.dim_out = channels // 2 + int(metadata)
        self.metadata = metadata

    def _make_layer(
        self,
        repeat_times: int,
        channels: int,
        block_type: type[_BasicBlock] = _BasicBlock,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.in_channels != channels * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    channels * block_type.expansion,
                    1,
                    stride,
                    0,
                    bias=False,
                ),
                nn.BatchNorm1d(channels * block_type.expansion),
            )

        layers = [
            block_type(
                self.in_channels,
                channels,
                stride,
                downsample,
                self.groups,
                self.base_channels,
            )
        ]

        self.in_channels = channels * block_type.expansion
        for _ in range(1, repeat_times):
            layers.append(block_type(self.in_channels, channels, 1, None, self.groups, self.base_channels))

        return nn.Sequential(*layers)

    def tokenize(self, profile: Tensor | Iterable[Tensor]) -> Dict[str, Tensor]:
        if not isinstance(profile, (list, tuple)):
            profile = [profile]
        profile = nn.utils.rnn.pad_sequence(profile, batch_first=True)
        return {"profile": profile}

    def forward_features(self, profile: Tensor) -> Tensor:
        profile = profile.transpose(1, 2)
        out = self.conv1(profile)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        for stage in self.stages:
            out = stage(out)

        return out

    def forward(self, profile: Tensor, **kwargs) -> Tensor:
        out = self.forward_features(profile)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        if self.metadata:
            metadata = kwargs["profile_len"].to(profile.dtype)
            metadata /= profile.shape[1]
            out = torch.cat((out, metadata), 1)

        return self.drop(out)
