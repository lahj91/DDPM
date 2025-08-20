"""
CelebA 데이터셋으로 DDPM 모델을 학습시키는 메인 스크립트
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from net import UNet
from diffusion import Diffusion

# 하이퍼파라미터 설정
IMG_SIZE = 64
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 100
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "./data/celeba"
OUTPUT_PATH = "./results"
CHECKPOINT_PATH = "./checkpoints"

def get_data():
    """
    CelebA 데이터셋을 불러오고 DataLoader를 생성합니다.
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # [-1, 1]로 정규화
    ])
    
    # 데이터셋 다운로드 (최초 실행 시 시간이 걸릴 수 있습니다)
    dataset = datasets.CelebA(root=DATA_PATH, split='all', download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    return dataloader

def train(model, diffusion, dataloader, optimizer, loss_fn, epoch):
    """
    한 에폭(epoch) 동안의 학습을 진행합니다.
    """
    model.train()
    pbar = tqdm(dataloader)
    total_loss = 0
    
    for i, (images, _) in enumerate(pbar):
        images = images.to(DEVICE)
        
        # 1. 랜덤 타임스텝 t 샘플링
        t = torch.randint(0, TIMESTEPS, (images.shape[0],)).to(DEVICE)
        
        # 2. 노이즈 생성 및 Forward Process를 통해 x_t (noisy image) 생성
        noise = torch.randn_like(images).to(DEVICE)
        x_t = diffusion.q_sample(x_0=images, t=t, noise=noise)
        
        # 3. 모델을 통해 노이즈 예측
        predicted_noise = model(x_t, t)
        
        # 4. Loss 계산 (실제 노이즈와 예측된 노이즈의 차이)
        loss = loss_fn(noise, predicted_noise)
        
        # 5. 역전파 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_description(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Average Loss: {avg_loss:.4f}")

def sample_images(model, diffusion, epoch, n_images=16):
    """
    현재 모델 상태에서 이미지를 샘플링하여 저장합니다.
    """
    model.eval()
    shape = (n_images, 3, IMG_SIZE, IMG_SIZE)
    imgs = diffusion.p_sample_loop(model, shape)
    
    # [-1, 1] 범위를 [0, 1]로 변환하여 저장
    final_img = (imgs[-1] + 1) * 0.5
    save_image(final_img, os.path.join(OUTPUT_PATH, f"sample_{epoch+1:03d}.png"), nrow=4)
    print(f"Saved sample images for epoch {epoch+1}")

def main():
    # 출력 및 체크포인트 디렉토리 생성
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # 모델, Diffusion, 데이터로더, 옵티마이저, 손실 함수 초기화
    model = UNet().to(DEVICE)
    diffusion = Diffusion(timesteps=TIMESTEPS, img_size=IMG_SIZE, device=DEVICE)
    dataloader = get_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    print("Training started...")
    for epoch in range(EPOCHS):
        train(model, diffusion, dataloader, optimizer, loss_fn, epoch)
        
        # 주기적으로 샘플 이미지 생성 및 모델 저장
        if (epoch + 1) % 5 == 0:
            sample_images(model, diffusion, epoch)
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f"ddpm_celeba_epoch_{epoch+1}.pth"))
            print(f"Saved model checkpoint for epoch {epoch+1}")

if __name__ == "__main__":
    main()