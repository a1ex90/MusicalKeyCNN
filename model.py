import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    """
    Basic convolutional block as used in the architecture

    Each block consists of a 2D convolution, followed by batch normalization and ELU activation.
    This design allows for stable and efficient learning on log-magnitude log-frequency audio spectrograms,
    preserving spectro-temporal information relevant for musical key detection.
    
    Args:
        in_channels (int): Number of input channels (e.g., 1 for single-channel spectrograms).
        out_channels (int): Number of output feature maps (controls model capacity).
        kernel_size (int or tuple): Size of the convolutional kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # 'same' padding ensures the feature map size is preserved, as described for the model
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding='same',
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        """
        Forward pass through the convolutional block.
        
        Args:
            x (torch.Tensor): Input spectrogram (B, C, F, T)

        Returns:
            torch.Tensor: Output after convolution, normalization, ELU.
        """
        x = self.conv(x)   # Linear time-frequency filtering
        x = self.bn(x)     # Stabilizes training across varying audio excerpts
        x = self.elu(x)    # Non-linearity for improved feature learning
        return x

class KeyNet(nn.Module):
    """
    Convolutional neural network for musical key classification, as introduced in
    Korzeniowski & Widmer (2018). This architecture is designed to generalize across musical genres
    by using only convolutional and pooling layers, omitting dense layers to reduce overfitting
    and allow for deeper, more expressive models.

    The network operates directly on log-frequency spectrogram snippets and outputs the predicted key class.
    Dropout is applied after pooling and at deeper layers to regularize the model, further improving
    generalization performance.

    Args:
        num_classes (int): Number of key classes, typically 24 (12 tonic * {major, minor}).
        in_channels (int): Input feature channels, usually 1 for log-magnitude spectrograms.
        Nf (int): Number of feature maps for the first convolution, controls model width/capacity.
        p (float): Dropout probability per section (2D feature map dropout).
    """
    def __init__(self, num_classes=24, in_channels=1, Nf=20, p=0.5):
        super().__init__()

        # Initial convolution, larger kernel for broader spectral context (5x5)
        self.conv1 = BasicConv2d(in_channels, Nf, kernel_size=5)
        # Subsequent convolution, smaller kernel (3x3)
        self.conv2 = BasicConv2d(Nf, Nf, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(p=p)

        # Second block, increased feature map count for deeper spectral representations
        self.conv3 = BasicConv2d(Nf, 2*Nf, kernel_size=3)
        self.conv4 = BasicConv2d(2*Nf, 2*Nf, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(p=p)

        # Third block, further channel doubling for higher-level abstraction
        self.conv5 = BasicConv2d(2*Nf, 4*Nf, kernel_size=3)
        self.conv6 = BasicConv2d(4*Nf, 4*Nf, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout2d(p=p)

        # Deep layers for complex pattern extraction and key decision
        self.conv7 = BasicConv2d(4*Nf, 8*Nf, kernel_size=3)
        self.dropout4 = nn.Dropout2d(p=p)
        self.conv8 = BasicConv2d(8*Nf, 8*Nf, kernel_size=3)
        self.dropout5 = nn.Dropout2d(p=p)

        # 1x1 convolution as the final classifier for key classes (no dense layer)
        self.conv9 = BasicConv2d(8*Nf, num_classes, kernel_size=1)
        # Global average pooling creates a fixed-size output across variable input durations
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Forward propagation through the KeyNet.

        Args:
            x (torch.Tensor): Input tensor (B, in_channels, freq_bins, time_frames)

        Returns:
            torch.Tensor: Output logits for each key class (B, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.conv7(x)
        x = self.dropout4(x)
        x = self.conv8(x)
        x = self.dropout5(x)
        x = self.conv9(x)

        # Global (frequency, time) averaging, as in Table 1b in the paper
        x = self.global_avgpool(x)
        x = torch.flatten(x, 1)
        return x

if __name__ == "__main__":
    # Example usage: dummy input for the key classification
    model = KeyNet(num_classes=24, in_channels=1)
    dummy_input = torch.randn(2, 1, 105, 100)  # (Batch, Channel, Frequency, Time)
    out = model(dummy_input)
    print("Shape:", out.shape)  # Expected: [2, 24]
    # Print number of trainable parameters for reproducibility/comparison
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))