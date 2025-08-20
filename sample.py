"""
학습된 DDPM 모델을 불러와 이미지를 생성하는 스크립트
"""
import os
import torch
from torchvision.utils import save_image

from net import UNet
from diffusion import Diffusion

# 설정
IMG_SIZE = 64
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_IMAGES = 16 # 생성할 이미지 개수
CHECKPOINT_FILE = "./checkpoints/ddpm_celeba_epoch_100.pth" # 불러올 모델 파일 경로
OUTPUT_FILE = "./generated_image.png"

def generate():
    # 모델 및 Diffusion 초기화
    model = UNet().to(DEVICE)
    
    # 학습된 가중치 불러오기
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"Error: Checkpoint file not found at {CHECKPOINT_FILE}")
        return
        
    model.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    diffusion = Diffusion(timesteps=TIMESTEPS, img_size=IMG_SIZE, device=DEVICE)

    # 이미지 생성
    print(f"Generating {N_IMAGES} images...")
    shape = (N_IMAGES, 3, IMG_SIZE, IMG_SIZE)
    imgs = diffusion.p_sample_loop(model, shape)
    
    # 최종 이미지를 [0, 1] 범위로 변환하여 저장
    final_img = (imgs[-1] + 1) * 0.5
    save_image(final_img, OUTPUT_FILE, nrow=4)
    
    print(f"Image saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate()
