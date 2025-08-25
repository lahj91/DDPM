"""
DDPM의 Forward/Reverse Process를 정의하는 스크립트
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """
    선형으로 beta 값을 스케줄링합니다.
    """
    return torch.linspace(start, end, timesteps)

class Diffusion:
    def __init__(self, timesteps: int = 1000, img_size: int = 64, device: str = "cuda") -> None:
        self.timesteps = timesteps
        self.img_size = img_size
        self.device = device
        
        # Beta 스케줄 및 관련 변수들 사전 계산
        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # q(x_t | x_0) 계산에 필요한 변수들
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # p(x_{t-1} | x_t, x_0) 계산에 필요한 변수들
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def _get_index_from_list(self, vals: Tensor, t: list, x_shape: tuple) -> Tensor:
        """
        특정 timestep t에 해당하는 값을 가져와서 이미지 배치 차원에 맞게 변환
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_0: Tensor, t: int, noise=None) -> Tensor:
        """
        Forward Process: x_0에서 x_t를 샘플링
        x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self._get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        mean = sqrt_alphas_cumprod_t * x_0
        variance = sqrt_one_minus_alphas_cumprod_t * noise
        
        return mean + variance

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: Tensor, t: int, t_index: list) -> Tensor:
        """
        Reverse Process: 모델을 이용해 x_t에서 x_{t-1}을 샘플링
        """
        betas_t = self._get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._get_index_from_list(torch.sqrt(1.0 / self.alphas), t, x.shape)
        
        # 모델을 통해 평균(mean) 계산
        predicted_noise = model(x, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            # 노이즈 추가
            posterior_variance_t = self._get_index_from_list(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape: tuple) -> Tensor:
        """
        전체 샘플링 루프: T부터 0까지 진행
        """
        device = next(model.parameters()).device
        b = shape[0]
        # T 시점의 순수 노이즈에서 시작
        img = torch.randn(shape, device=device)
        imgs = []

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
            imgs.append(img.cpu())
        return imgs

if __name__ == '__main__':
    # Diffusion 프로세스 테스트
    diffusion = Diffusion(timesteps=1000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_0 = torch.randn(4, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (4,)).to(device)
    x_t = diffusion.q_sample(x_0, t)
    print(f"Original shape: {x_0.shape}")
    print(f"Noisy shape: {x_t.shape}")