"""
Training of Compositional GAN on Custom dataset with Discriminator
and Generator imported from model.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from model import SpatialTransformerNetwork, Discriminator_STN, initialize_weights
import itertools
import Config

transform = transforms.Compose([
    transforms.Resize(Config.image_size),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(Config.channels)], [0.5 for _ in range(Config.channels)]
    ),
])

# Object a dataset
a_dataset = datasets.ImageFolder(root="a", transform=transform)
a_loader = DataLoader(a_dataset, batch_size=Config.batch_size, shuffle=True)
# Object b dataset
b_dataset = datasets.ImageFolder(root="b", transform=transform)
b_loader = DataLoader(b_dataset, batch_size=Config.batch_size, shuffle=True)
# Composition dataset
logo_dataset = datasets.ImageFolder(root="composition", transform=transform)
logo_loader = DataLoader(logo_dataset, batch_size=Config.batch_size, shuffle=True)

# Spatial Transformer Network
Comp_STN_Disc = Discriminator_STN(Config.channels).to(Config.device)
Decomp_STN_Disc = Discriminator_STN(Config.channels).to(Config.device)
Comp_STN = SpatialTransformerNetwork().to(Config.device)
Decomp_STN = SpatialTransformerNetwork().to(Config.device)

# initialize weights
initialize_weights(Comp_STN_Disc)
initialize_weights(Decomp_STN_Disc)
initialize_weights(Comp_STN)
initialize_weights(Decomp_STN)

opt_Disc_STN = optim.Adam(itertools.chain(Comp_STN_Disc.parameters(), Decomp_STN_Disc.parameters()),
                          lr=Config.lr, betas=(0.5, 0.999))
opt_STN = optim.SGD(itertools.chain(Comp_STN.parameters(), Decomp_STN.parameters()),
                     lr=0.001)

L1 = nn.L1Loss()
mse = nn.MSELoss()

D_scaler = torch.cuda.amp.GradScaler()
STN_scaler = torch.cuda.amp.GradScaler()

Comp_STN_Disc.train()
Decomp_STN_Disc.train()
Comp_STN.train()
Decomp_STN.train()

steps = 0
img_num = 0
print('start training compositional model')
for epoch in range(1, Config.num_epochs + 1):
    for (batch_idx, (logo, _)) in enumerate(logo_loader):
        for (_, (a, _)), (_, (b, _)) in zip(enumerate(a_loader), enumerate(b_loader)):
            torch.cuda.empty_cache()

            a = a.to(Config.device)
            b = b.to(Config.device)
            logo = logo.to(Config.device)

            # Train Spatial Transformer Network Discriminator
            # Comp
            a_b = torch.cat([a, b], 1).to(Config.device)
            a_T, b_T = Comp_STN(a_b)

            Disc_STN_logo_T = Comp_STN_Disc(logo)
            Disc_STN_a_T = Comp_STN_Disc(a_T)
            Disc_STN_b_T = Comp_STN_Disc(b_T)

            Disc_STN_logo_T_loss = mse(Disc_STN_logo_T, torch.ones_like(Disc_STN_logo_T))
            Disc_STN_a_T_loss = mse(Disc_STN_a_T, -torch.ones_like(Disc_STN_a_T))
            Disc_STN_b_T_loss = mse(Disc_STN_b_T, -torch.ones_like(Disc_STN_b_T))

            Disc_Comp_STN_loss = (
                    Disc_STN_logo_T_loss +
                    Disc_STN_a_T_loss +
                    Disc_STN_b_T_loss
            )

            # Decomp
            a_T_b_T = torch.cat([a_T, b_T], 1).to(Config.device)
            a_rT, b_rT = Decomp_STN(a_T_b_T)

            Disc_STN_a = Decomp_STN_Disc(a)
            Disc_STN_b = Decomp_STN_Disc(b)
            Disc_STN_a_rT = Decomp_STN_Disc(a_rT)
            Disc_STN_b_rT = Decomp_STN_Disc(b_rT)

            Disc_STN_a_rT_real_loss = mse(Disc_STN_a, torch.ones_like(Disc_STN_a))
            Disc_STN_b_rT_real_loss = mse(Disc_STN_b, torch.ones_like(Disc_STN_b))
            Disc_STN_a_rT_fake_loss = mse(Disc_STN_a_rT, -torch.ones_like(Disc_STN_a_rT))
            Disc_STN_b_rT_fake_loss = mse(Disc_STN_b_rT, -torch.ones_like(Disc_STN_b_rT))

            Disc_Decomp_STN_loss = (
                    Disc_STN_a_rT_real_loss +
                    Disc_STN_a_rT_fake_loss +
                    Disc_STN_b_rT_real_loss +
                    Disc_STN_b_rT_fake_loss
            )

            # put it together
            D_loss = (Disc_Comp_STN_loss + Disc_Decomp_STN_loss) / 2

            opt_Disc_STN.zero_grad()
            D_scaler.scale(D_loss).backward(retain_graph=True)
            D_scaler.step(opt_Disc_STN)
            D_scaler.update()

            # Train Spatial Transformer Network
            D_a_T_fake = Comp_STN_Disc(a_T)
            D_b_T_fake = Comp_STN_Disc(b_T)
            STN_a_T_loss = mse(D_a_T_fake, torch.ones_like(D_a_T_fake))
            STN_b_T_loss = mse(D_b_T_fake, torch.ones_like(D_b_T_fake))

            D_a_rT_fake = Decomp_STN_Disc(a_rT)
            D_b_rT_fake = Decomp_STN_Disc(b_rT)
            STN_a_rT_loss = mse(D_a_rT_fake, torch.ones_like(D_a_rT_fake))
            STN_b_rT_loss = mse(D_b_rT_fake, torch.ones_like(D_b_rT_fake))

            loss_G = (
                STN_a_T_loss +
                STN_b_T_loss +
                STN_a_rT_loss +
                STN_b_rT_loss
            )

            # cycle loss
            a_rT_b_rT = torch.cat([a_rT, b_rT], 1).to(Config.device)
            cycle_a_T, cycle_b_T = Comp_STN(a_rT_b_rT)
            cycle_a_T_cycle_b_T = torch.cat([cycle_a_T, cycle_b_T], 1).to(Config.device)
            cycle_a_rT, cycle_b_rT = Decomp_STN(cycle_a_T_cycle_b_T)

            cycle_a_T_loss = L1(a_T, cycle_a_T)
            cycle_a_rT_loss = L1(a, cycle_a_rT)
            cycle_b_T_loss = L1(b_T, cycle_b_T)
            cycle_b_rT_loss = L1(b, cycle_b_rT)

            # L1 loss
            l1_loss = L1(a_T, logo) + L1(b_T, logo)

            cycle_loss = (
                cycle_a_T_loss +
                cycle_a_rT_loss +
                cycle_b_T_loss +
                cycle_b_rT_loss
            )

            STN_loss = (
                loss_G +
                50 * cycle_loss
            )

            opt_STN.zero_grad()
            STN_scaler.scale(STN_loss).backward()
            STN_scaler.step(opt_STN)
            STN_scaler.update()

            if batch_idx % 100 == 0:
                save_image(logo * 0.5 + 0.5, f"saved_images/STN_training/logo/logo_{img_num}.png")
                save_image(a_T * 0.5 + 0.5, f"saved_images/STN_training/a_T/a_T_{img_num}.png")
                save_image(b_T * 0.5 + 0.5, f"saved_images/STN_training/b_T/b_T_{img_num}.png")
                save_image(a_rT * 0.5 + 0.5, f"saved_images/STN_training/a_rT/a_T_{img_num}.png")
                save_image(b_rT * 0.5 + 0.5, f"saved_images/STN_training/b_rT/b_T_{img_num}.png")

                img_num += 1

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0:
                print(
                    f'Epoch [{epoch:2d}/{Config.num_epochs}] Batch {batch_idx:4d}/{len(logo_loader)}   '
                    f'Loss D_STN: {D_loss:8.4f}, Loss STN: {STN_loss:8.4f} , img num: {img_num-1}'
                )
    # Save STN model
    torch.save(Comp_STN, './saved_model/Comp_STN_Model.pt')
    torch.save(Decomp_STN, './saved_model/Decomp_STN_Model.pt')


# Main Function
# if Config.train_stn:
#     train(Config.stn_iterations)
# else:
#     STN = torch.load('./saved_model/STN_Model.pt')
#     STN = STN.to(Config.device)
