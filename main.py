import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from models import RRDBNet, Discriminator, initialize_weights
from loss import VGGLoss
from utils import gradient_penalty, plot_examples, save_checkpoint, load_checkpoint, PSNR
from torch.utils.tensorboard import SummaryWriter
from dataset import get_data_loaders
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


BATCH_SIZE = 1
LR = 0.001
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data/QXSLAB_SAROPT"
MODALITIES = ["sar_compact", "opt_compact"]
LAMBDA_GP = 10

SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth"
CHECKPOINT_DISC = "disc.pth"

amp_enabled = True
lambda_value = 10.0

def main():
    g_scaler = torch.amp.GradScaler('cuda')
    d_scaler = torch.amp.GradScaler('cuda')

    vgg_loss = VGGLoss()
    l1 = nn.L1Loss()

    # Load Data
    train_loader, test_loader = get_data_loaders(DATA_DIR, MODALITIES, BATCH_SIZE)

    print("Finished loading data.")

    # Define model
    generator = RRDBNet(1, 3, 64, 23, gc=32)
    discriminator = Discriminator(3)

    generator = generator.to(DEVICE)
    initialize_weights(generator)
    discriminator = discriminator.to(DEVICE)

    optim_g = optim.Adam(generator.parameters(), lr=LR, betas=(0.9, 0.999))
    optim_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.9, 0.999))

    writer = SummaryWriter("logs")
    tb_step = 0

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader, leave=True)
        for idx, batch in enumerate(loop):
            generator.train()
            discriminator.train()

            sar_img, opt_img = batch[MODALITIES[0]], batch[MODALITIES[1]]
            sar_img, opt_img = sar_img.to(DEVICE), opt_img.to(DEVICE)

            with torch.amp.autocast('cuda'):
                fake = generator(sar_img)
                critic_real = discriminator(opt_img)
                critic_fake = discriminator(fake.detach())
                gp = gradient_penalty(discriminator, opt_img, fake, device=DEVICE)
                loss_critic = (
                        -(torch.mean(critic_real) - torch.mean(critic_fake))
                        + LAMBDA_GP * gp
                )

                img_psnr = PSNR(fake, sar_img)
                img_mse = F.mse_loss(fake, sar_img)
                img_ssim = ssim(sar_img, fake, data_range=fake.max() - fake.min())
                print(f'MSE: {img_mse:.2f}, SSIM: {img_ssim:.2f}, PSNR: {img_psnr:.2f}')

            optim_d.zero_grad()
            d_scaler.scale(loss_critic).backward()
            d_scaler.step(optim_d)
            d_scaler.update()

            # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            with torch.amp.autocast('cuda'):
                l1_loss = 1e-2 * l1(fake, opt_img)
                adversarial_loss = 5e-3 * -torch.mean(discriminator(fake))
                loss_for_vgg = vgg_loss(fake, opt_img)
                gen_loss = l1_loss + loss_for_vgg + adversarial_loss

            optim_g.zero_grad()
            g_scaler.scale(gen_loss).backward()
            g_scaler.step(optim_g)
            g_scaler.update()

            writer.add_scalar("Critic loss", loss_critic.item(), global_step=tb_step)
            tb_step += 1

            # if idx % 500 == 0 and idx > 0:
            #     plot_examples("test_images/", generator)

            loop.set_postfix(
                gp=gp.item(),
                critic=loss_critic.item(),
                l1=l1_loss.item(),
                vgg=loss_for_vgg.item(),
                adversarial=adversarial_loss.item(),
            )

            if SAVE_MODEL:
                save_checkpoint(generator, optim_g, filename=CHECKPOINT_GEN)
                save_checkpoint(discriminator, optim_d, filename=CHECKPOINT_DISC)



if __name__ == "__main__":
    try_model = False

    if try_model:
        # Will just use pretrained weights and run on images
        # in test_images/ and save the ones to SR in saved/
        gen = RRDBNet(1, 3, 64, 23).to(DEVICE)
        opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.0, 0.9))
        load_checkpoint(
            CHECKPOINT_GEN,
            gen,
            opt_gen,
            LR,
        )
        plot_examples(os.path.join(DATA_DIR, "test_images/"), gen)
    else:
        # This will train from scratch
        main()
