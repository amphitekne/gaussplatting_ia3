import torch


def PSNR(image: torch.Tensor, gt_image: torch.Tensor, device="cuda"):
    mse = torch.mean((gt_image.to(device) - image.to(device)) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


def MSE(image: torch.Tensor, gt_image: torch.Tensor):
    return ((image - gt_image) ** 2).mean()
