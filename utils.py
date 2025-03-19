import torch
import os
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.metrics import structural_similarity as ssim


test_transform = A.Compose(
        [
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2(),
        ]
    )

# test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data/QXSLAB_SAROPT"

def PSNR(predicted, target, max_pixel_value=1.0):
    mse = F.mse_loss(predicted, target)
    psnr_value = 10 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr_value


def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def plot_examples(low_res_folder, gen):
    files = os.listdir(low_res_folder)

    gen.eval()
    for file in tqdm(files):
        image = Image.open(os.path.join(DATA_DIR, "test_images/") + file)
        image = test_transform(image)
        with torch.no_grad():
            upscaled_img = gen(
                image
                .unsqueeze(0)
                .to(DEVICE)
            )

            img_psnr = PSNR(upscaled_img, image)
            img_mse = F.mse_loss(upscaled_img, image)
            img_ssim = ssim(image, upscaled_img, data_range=upscaled_img.max() - upscaled_img.min())
            print(f'MSE: {img_mse:.2f}, SSIM: {img_ssim:.2f}, PSNR: {img_psnr:.2f}')

        save_image(upscaled_img, f"{os.path.join(DATA_DIR, "saved")}/{file}")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    # model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
