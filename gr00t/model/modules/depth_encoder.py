import torch
import torch.nn as nn
import torchvision.models as models

class DepthEncoder(nn.Module):
    """
    A lightweight depth encoder based on ResNet18.
    It takes a single-channel depth image and outputs a feature vector.
    """
    def __init__(self, output_dim=2048, pretrained=True):
        super().__init__()
        # Load a lightweight ResNet (ResNet18)
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modify the first layer to accept 1-channel input (depth) instead of 3 (RGB)
        # We can sum the weights of the RGB channels to initialize the single channel weights
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # Initialize the new conv1 weights
        with torch.no_grad():
            self.backbone.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)
            
        # Remove the fully connected layer (fc) to get features
        # ResNet18's last conv layer output is 512 channels
        self.feature_dim = 512
        self.backbone.fc = nn.Identity()
        
        # Project to the desired output dimension
        self.projector = nn.Linear(self.feature_dim, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: Depth images of shape (B, 1, H, W) or (B, T, 1, H, W)
        Returns:
            Features of shape (B, output_dim) or (B, T, output_dim)
        """
        is_sequence = x.ndim == 5
        if is_sequence:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            
        # Forward pass through ResNet backbone
        features = self.backbone(x)
        
        # Project to output dimension
        features = self.projector(features)
        
        if is_sequence:
            features = features.view(B, T, -1)
            
        return features
