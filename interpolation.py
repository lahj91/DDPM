import torch
import torchvision
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import os
from net import UNet
from diffusion import Diffusion

IMG_SIZE = 64
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "./data/celeba"
CHECKPOINT_PATH = "./checkpoints"
IMG_SIZE = 64
TIMESTEPS = 1000
CHECKPOINT_FILE = "./checkpoints/ddpm_celeba_epoch_100.pth" # 불러올 모델 파일 경로
SAVE_PATH = "./results/interpolation_result.png"
save_dir = os.path.dirname(SAVE_PATH)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Directory '{save_dir}' created.")

def plot_grid(tensor, figsize=(12, 12), title=None):
    """생성된 이미지 그리드를 시각화하는 헬퍼 함수"""
    grid = tensor.cpu().clone().detach()
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=figsize)
    plt.imshow(grid)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=16)
    plt.show()

def generate_interpolation_grid(model, diffusion, img1, img2, t_steps, lambdas, device):
    """논문의 Figure 9와 같은 보간 그리드를 생성합니다."""
    model.eval()
    img1, img2 = img1.to(device).unsqueeze(0), img2.to(device).unsqueeze(0)
    
    all_rows_tensors = []

    for t_start in t_steps:
        print(f"Processing for t = {t_start}...")
        
        if t_start == 0:
            row_images = [img1, img1] # Source, Reconstruction.1
            for lam in lambdas:
                interp_img = (1 - lam) * img1 + lam * img2
                row_images.append(interp_img)
            row_images.extend([img2, img2]) # Reconstruction.2, Source
            all_rows_tensors.append(torch.cat(row_images, dim=0))
            continue

        t = torch.full((1,), t_start, device=device, dtype=torch.long)
        noise = torch.randn_like(img1, device=device)
        
        x_t1 = diffusion.q_sample(x_0=img1, t=t, noise=noise)
        x_t2 = diffusion.q_sample(x_0=img2, t=t, noise=noise)

        row_images = [img1] # Source 1

        # Denoising loop를 별도 함수로 만들어 재사용
        def denoise(start_img, start_t):
            img = start_img
            for i in reversed(range(0, start_t)):
                t_step = torch.full((1,), i, device=device, dtype=torch.long)
                img = diffusion.p_sample(model, img, t_step, i)
            return img

        # Rec. 1 생성
        row_images.append(denoise(x_t1, t_start))

        # Lambda 값에 따라 보간 및 Denoising
        for lam in lambdas:
            x_interp_t = (1 - lam) * x_t1 + lam * x_t2
            row_images.append(denoise(x_interp_t, t_start))

        # Rec. 2 생성 및 Source 2 추가
        row_images.append(denoise(x_t2, t_start))
        row_images.append(img2)
        
        all_rows_tensors.append(torch.cat(row_images, dim=0))

    full_tensor = torch.cat(all_rows_tensors, dim=0)
    
    # 그리드의 한 행에 들어갈 이미지 수
    num_cols = 2 + len(lambdas) + 2 
    grid = torchvision.utils.make_grid(full_tensor, nrow=num_cols, normalize=True, scale_each=True)
    return grid



def run_full_interpolation():
    """모델 로드부터 그리드 생성 및 시각화까지 전체 과정을 실행합니다."""
    # 모델 및 Diffusion 초기화
    model = UNet().to(DEVICE)
    
    # 학습된 가중치 불러오기
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"Error: Checkpoint file not found at {CHECKPOINT_FILE}")
        return
    model.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=DEVICE))
    print("Model loaded successfully.")

    diffusion = Diffusion(timesteps=TIMESTEPS, img_size=IMG_SIZE, device=DEVICE)
    
    # 데이터 로드
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CelebA(root=DATA_PATH, split='test', download=True, transform=transform)
    
    # 보간에 사용할 두 이미지 선택 (예: 0번, 100번 이미지)
    img1, _ = dataset[0]
    img2, _ = dataset[100]

    # Figure 9와 유사한 t 단계 설정
    t_steps_to_show = [999, 875, 750, 625, 500, 375, 250, 125, 0]
    
    # Figure 9와 동일한 lambda 값 설정 (0.1 ~ 0.9)
    lambdas_to_show = np.linspace(0.1, 0.9, 9)

    # 그리드 생성
    grid = generate_interpolation_grid(model, diffusion, img1, img2, t_steps_to_show, lambdas_to_show, DEVICE)
    
    save_image(grid, SAVE_PATH)
    print(f"Interpolation grid saved to .{SAVE_PATH}")
    # # 결과 시각화
    # plot_grid(grid, title="Coarse-to-fine interpolations (DDPM)")

if __name__ == "__main__":
    run_full_interpolation()