"""
DDPM에서 노이즈 예측을 위해 사용될 U-Net 모델 아키텍처
참고: https://github.com/lucidrains/denoising-diffusion-pytorch
"""
import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Transformer에서 영감을 받은 시간(t)에 대한 Sinusoidal Position Embedding
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 + Time Embedding"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, t):
        # Time embedding을 더하기 전에 먼저 convolution을 적용합니다.
        h = self.double_conv(x)
        
        # 시간 임베딩 주입
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(..., ) + (None, ) * 2] # (B, C) -> (B, C, 1, 1)
        return h + time_emb

class UNet(nn.Module):
    """
    DDPM을 위한 표준 U-Net 모델
    """
    def __init__(self, img_channels=3, time_emb_dim=256):
        super().__init__()
        
        # 시간 임베딩
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Downsampling path (Contracting path)
        self.down1 = DoubleConv(img_channels, 64, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256, time_emb_dim)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512, time_emb_dim)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bot = DoubleConv(512, 1024, time_emb_dim)

        # Upsampling path (Expansive path)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1 = DoubleConv(1024, 512, time_emb_dim) # 512 (skip) + 512 (up) = 1024
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = DoubleConv(512, 256, time_emb_dim) # 256 (skip) + 256 (up) = 512
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = DoubleConv(256, 128, time_emb_dim) # 128 (skip) + 128 (up) = 256
        
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4 = DoubleConv(128, 64, time_emb_dim) # 64 (skip) + 64 (up) = 128
        
        # Final output layer
        self.output = nn.Conv2d(64, img_channels, kernel_size=1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        
        # Downsampling
        d1 = self.down1(x, t)
        p1 = self.pool1(d1)
        
        d2 = self.down2(p1, t)
        p2 = self.pool2(d2)
        
        d3 = self.down3(p2, t)
        p3 = self.pool3(d3)
        
        d4 = self.down4(p3, t)
        p4 = self.pool4(d4)
        
        # Bottleneck
        b = self.bot(p4, t)
        
        # Upsampling & Skip-connection
        u1 = self.upconv1(b)
        cat1 = torch.cat([u1, d4], dim=1) # skip connection
        u1_conv = self.up1(cat1, t)
        
        u2 = self.upconv2(u1_conv)
        cat2 = torch.cat([u2, d3], dim=1) # skip connection
        u2_conv = self.up2(cat2, t)
        
        u3 = self.upconv3(u2_conv)
        cat3 = torch.cat([u3, d2], dim=1) # skip connection
        u3_conv = self.up3(cat3, t)
        
        u4 = self.upconv4(u3_conv)
        cat4 = torch.cat([u4, d1], dim=1) # skip connection
        u4_conv = self.up4(cat4, t)
        
        return self.output(u4_conv)

if __name__ == '__main__':
    # 모델 테스트
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    x = torch.randn(4, 3, 64, 64).to(device)
    t = torch.randint(1, 1000, (4,)).to(device)
    predicted_noise = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Predicted noise shape: {predicted_noise.shape}")
