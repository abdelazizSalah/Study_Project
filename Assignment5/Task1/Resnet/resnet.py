'''
@Author: Abdelaziz Neamatallah
@Date: 13.1.26
@Description: This file contains the implementation for ResNet-18 architecture using PyTorch.

1. Residual block (basicBlock 1D)
   1. 2 convolution layers per block:
      - Conv1D -> Batch Normalization (BN) -> ReLU -> conv1D -> BN     
   2. Skip connection:
      1. identity if shapes match
      2. projection (1x1 Conv1d) if channels/stride changes
2. ResNet-18 stages
   - it should consist of 4 stages, each stage has 2 residual blocks
   - Standard channel progression example: 
     1. Stage1: 64
     2. Stage2: 128
     3. Stage3: 256
     4. Stage4: 512
   - Total "18 Layers" requirement: 
     - follow the ResNet-18 pattern (counts alogn when we consider convs)
3. Pooling
   - We can use AdaptiveAvgPool1d(1) at the end
     - it gives fixed-size embedding regardless of M
4. 2 Fully connected layers
   1. FC1 (embeddings -> hidden) + ReLU + DropOut
   2. FC2 (hidden -> 2 logits)
5. Sanity checks
   - we can run on one batch to check:
     - shapes
     - logits shape
     - loss runs

'''

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1) BasicBlock (1D) for ResNet-18
# -----------------------------
class BasicBlock1D(nn.Module):
    """
    ResNet-18 BasicBlock: two Conv1d layers with BN + ReLU between,
    plus an identity/projection skip connection.
    """

    # no expansion for the output channels in BasicBlock
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        '''
        in_channels:  number of input channels
        out_channels: number of output channels
        stride: controls downsampling
        
        '''
        # initialize parent class
        super().__init__()

        # first conv layer with kernel size = 3 and padding = 1 to maintain the sequence length
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )

        # normalizes each channel across the batch and sequence length
        # it improves training stability and convergence
        self.bn1 = nn.BatchNorm1d(out_channels)

        # second conv layer, forces stride = 1 to perserve resolution inside the block
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        # second batch normalization layer
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip path: identity if shape matches, otherwise 1x1 projection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        '''
            Forward pass of the BasicBlock1D.
            x: input tensor of shape (B, C_in, M)
            returns: output tensor of shape (B, C_out, M_out)
        
        '''

        # stores the input for skip connection
        identity = x

        # First conv + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        # Second conv + BN, and we do not use ReLU here yet
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        # applies skip connection to enable the residual learning
        out = out + identity
        out = F.relu(out, inplace=True)
        return out


# -----------------------------
# 2) ResNet-18 backbone (1D): 4 stages, each stage has 2 blocks
# -----------------------------
class ResNet18_1D(nn.Module):
    """
    ResNet-18 (1D) layout:
      stem: Conv1d(7) + BN + ReLU + MaxPool
      stage1: 64  (2 blocks)
      stage2: 128 (2 blocks, first block stride=2)
      stage3: 256 (2 blocks, first block stride=2)
      stage4: 512 (2 blocks, first block stride=2)
      head: AdaptiveAvgPool1d(1) -> FC1+ReLU+Dropout -> FC2 (2 logits)

    Optionally concatenates 5 stats features at the embedding stage.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout_p: float = 0.3,
        use_stats: bool = False, # whether to use the 5 stats features
        stats_dim: int = 5, # number of stats features (if used)
    ):
        super().__init__()
        self.use_stats = use_stats
        self.stats_dim = stats_dim

        # Stem (like classic ResNet, adapted to 1D)
        '''
            Stem are the very first part of the network, applied directly to the raw input
            its main purpose is to: 
                - Quickly extract basic low-level patterns
                - Reduce the input resolution early
                - Convert raw input into a feature representation suitable for deep processing
            For input shape: (B, 1, M)
            after Stem: 
                (B, 64, M/4)
            So the number of channels increases, but the length M decreases to the quarter
        '''
        self.stem_conv = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm1d(64)
        self.stem_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 4 stages, 2 blocks each (ResNet-18 pattern)
        '''
            Stages are groups of residual blocks that form the main body of the ResNet
            Each stage: 
                - Has a fixed number of channels
                - Contains multiple residual blocks
                - Learns features at a specific abstraction level
            So we perform the following each stage:
                - Use Residual learning
                - Applies Conv -> BN -> ReLU -> Conv -> BN
                - Refines features rather than just detecting them
            Progression: 
                - Stage 1 -> low-level features
                - Stage 2 -> mid-level features
                - Stage 3 -> high-level features
                - Stage 4 -> very abstract features
        '''
        self.in_planes = 64
        self.layer1 = self._make_layer(out_channels=64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(out_channels=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(out_channels=256, blocks=2, stride=2)
        self.layer4 = self._make_layer(out_channels=512, blocks=2, stride=2)

        # Pooling -> embedding
        # convert (B, 512, M) to (B, 512, 1) independent of the input length M
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # FC head (2 FC layers)
        # embedding_dim = 512 after avgpool (because stage4 out_channels=512)
        embedding_dim = 512
        fc1_in = embedding_dim + (stats_dim if use_stats else 0)

        self.fc1 = nn.Linear(fc1_in, hidden_dim)
        self.drop = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # init like typical ResNet
        self._init_weights()

    def _make_layer(self, out_channels: int, blocks: int, stride: int):
        layers = []
        # First block may downsample via stride
        layers.append(BasicBlock1D(self.in_planes, out_channels, stride=stride))
        self.in_planes = out_channels * BasicBlock1D.expansion
        # Remaining blocks keep stride=1
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_planes, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        '''
            These are the standard normalization for ReLU and BatchNorm networks 
        
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, stats=None):
        """
        x:     (B, C_in, M)  e.g. (B,1,M) for raw bytes/sequence
        stats: (B, 5) if use_stats=True, else None
        """
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = F.relu(x, inplace=True)
        x = self.stem_pool(x)

        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Adaptive pooling -> (B, 512, 1) -> flatten -> (B, 512)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Optionally concatenate stats features
        if self.use_stats:
            if stats is None:
                raise ValueError("use_stats=True but stats=None was passed to forward().")
            if stats.dim() != 2 or stats.size(1) != self.stats_dim:
                raise ValueError(f"stats must be (B,{self.stats_dim}), got {tuple(stats.shape)}.")
            x = torch.cat([x, stats], dim=1)

        # FC head: FC1 + ReLU + Dropout, then FC2 -> logits
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.drop(x)
        logits = self.fc2(x)
        return logits


# -----------------------------
# 5) Sanity checks (shapes + logits + loss)
# -----------------------------
def sanity_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 8
    M = 256  # sequence length (<= your CLI M)
    x = torch.randn(B, 1, M).to(device)  # (B, C_in, M)

    # Without stats
    model = ResNet18_1D(in_channels=1, use_stats=False).to(device)
    logits = model(x)
    print("No-stats logits shape:", logits.shape)  # expected (B,2)

    y = torch.randint(0, 2, (B,), device=device)
    loss = F.cross_entropy(logits, y)
    print("No-stats loss:", float(loss))

    # With stats
    stats = torch.randn(B, 5).to(device)  # (B,5)
    model_stats = ResNet18_1D(in_channels=1, use_stats=True, stats_dim=5).to(device)
    logits2 = model_stats(x, stats=stats)
    print("With-stats logits shape:", logits2.shape)  # expected (B,2)

    loss2 = F.cross_entropy(logits2, y)
    print("With-stats loss:", float(loss2))


if __name__ == "__main__":
    sanity_check()
