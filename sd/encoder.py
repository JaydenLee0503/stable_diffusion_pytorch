import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

"""
class VAE_Encoder(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_res_blocks, attention_resolutions):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.res_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(VAE_ResidualBlock(out_channels, out_channels))
            if out_channels in attention_resolutions:
                self.res_blocks.append(VAE_AttentionBlock(out_channels))
        
        self.final_norm = nn.GroupNorm(32, out_channels)
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.final_norm(x)
        x = F.relu(x)
        x = self.final_conv(x)
        return x
"""
    
class VAE_EncoderfromScratch(nn.Sequential):
    def __init__(self):
        super().__init__()
        nn.Conv2d(3, 128, kernel_size=3, padding=1),
        VAE_ResidualBlock(128,128),
        VAE_ResidualBlock(128,128),
        # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height/2, Width/2)
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
        # (Batch_Size, 128, Height/2, Width/2) --> (Batch_Size, 256, Height/2, Width/2)
        VAE_ResidualBlock(128,256),
        # (Batch_Size, 256, Height/2, Width/2) --> (Batch_Size, 256, Height/2, Width/2)
        VAE_ResidualBlock(256,256),
        # (Batch_Size, 256, Height/2, Width/2) --> (Batch_Size, 256, Height/4, Width/4)
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
        # (Batch_Size, 256, Height/4, Width/4) --> (Batch_Size, 512, Height/4, Width/4)
        VAE_ResidualBlock(256,512),
        # (Batch_Size, 512, Height/4, Width/4) --> (Batch_Size, 512, Height/4, Width/4)
        VAE_ResidualBlock(512,512),
        # (Batch_Size, 512, Height/4, Width/4) --> (Batch_Size, 512, Height/8, Width/8)

        nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
        VAE_ResidualBlock(512,512),
        VAE_ResidualBlock(512,512),
        VAE_ResidualBlock(512,512),

        VAE_AttentionBlock(512),

        VAE_ResidualBlock(512,512),

        nn.GroupNorm(32, 512),
        nn.SiLU(),

        nn.Conv2d(512, 8, kernel_size=3, padding=1),
    
        nn.Conv2d(8, 8, kernel_size=1, padding=0),

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, Out_channels, Height/8, Width/8)
        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                x = F.pad(x, (0,1,0,1), mode='constant', value=0)
            x = module(x)
        # (Batch_size, 8, Height/8, Width/8) --> two tensors of shape (Batch_size, 4, Height, Height/8, Width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # (Batch_size, 4, Height/8, Width/8) --> (Batch_size, 4, Height, Height/8, Width/8)
        log_variance = torch.clamp(log_variance, min=-30.0, max=20.0)
        # (Batch_size, 4, Height/8, Width/8) --> (Batch_size, 4, Height, Height/8, Width/8)
        variance = log_variance.exp()
        # (Batch_size, 4, Height/8, Width/8) --> (Batch_size, 4, Height, Height/8, Width/8)
        stdev = torch.sqrt(variance)

        # N(0,1) --> N(mean, variance)=X?
        # X = mean + stdev * Z
        x = mean + stdev * noise

        # Scale the output by a constant
        x *= 0.18215

        return x
