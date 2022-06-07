"""
Training of Compositional GAN on Custom dataset with Discriminator
and Generator imported from model.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Discriminator_STN
from model import initialize_weights
from utils import save_checkpoint
import Config

transform = transforms.Compose([
    transforms.Resize([Config.image_size, Config.image_size]),
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
# Object c dataset
c_dataset = datasets.ImageFolder(root="c", transform=transform)
c_loader = DataLoader(c_dataset, batch_size=Config.batch_size, shuffle=True)
# Composition dataset
comp_dataset = datasets.ImageFolder(root="composition", transform=transform)
comp_loader = DataLoader(comp_dataset, batch_size=Config.batch_size, shuffle=True)

# Compositional Module
Discriminator = Discriminator_STN(Config.channels).to(Config.device)

opt_Disc_comp = optim.Adam(Discriminator.parameters(), lr=Config.lr, betas=(0.5, 0.999))

# initialize weights
initialize_weights(Discriminator)

L1 = nn.L1Loss()
mse = nn.MSELoss()

D_scaler = torch.cuda.amp.GradScaler()

Discriminator.train()

img_num = 0
print('Start Training Discriminator Model')
for epoch in range(1, 2):
    for (batch_idx, (real_comp, _)) in enumerate(comp_loader):
        for (_, (a, _)), (_, (b, _)), (_, (c, _)) in zip(enumerate(a_loader), enumerate(b_loader), enumerate(c_loader)):
            torch.cuda.empty_cache()

            a = a.to(Config.device)
            b = b.to(Config.device)
            c = c.to(Config.device)
            real_comp = real_comp.to(Config.device)

            # # # # # # # # # # # #
            # Train Discriminator #
            # # # # # # # # # # # #

            # Only Train Comp STN Disc
            # Decomp Disc Using L1 Loss
            Disc_STN_real_comp = Discriminator(real_comp)
            Disc_STN_a_T = Discriminator(a)
            Disc_STN_b_T = Discriminator(b)
            Disc_STN_c_T = Discriminator(c)
            Disc_STN_ones_comp = Discriminator(torch.ones_like(real_comp))
            Disc_STN_zero_comp = Discriminator(torch.zeros_like(real_comp))

            Disc_STN_real_comp_loss = mse(Disc_STN_real_comp, torch.ones_like(Disc_STN_real_comp))
            Disc_STN_a_T_loss = mse(Disc_STN_a_T, torch.zeros_like(Disc_STN_a_T))
            Disc_STN_b_T_loss = mse(Disc_STN_b_T, torch.zeros_like(Disc_STN_b_T))
            Disc_STN_c_T_loss = mse(Disc_STN_c_T, torch.zeros_like(Disc_STN_c_T))
            Disc_STN_ones_comp_loss = mse(Disc_STN_ones_comp, torch.zeros_like(Disc_STN_ones_comp))
            Disc_STN_zero_comp_loss = mse(Disc_STN_zero_comp, torch.zeros_like(Disc_STN_zero_comp))

            Disc_Comp_STN_loss = (
                Disc_STN_real_comp_loss +
                Disc_STN_a_T_loss +
                Disc_STN_b_T_loss +
                Disc_STN_c_T_loss +
                Disc_STN_ones_comp_loss +
                Disc_STN_zero_comp_loss
            )

            Discriminator.zero_grad()
            Disc_Comp_STN_loss.backward()
            opt_Disc_comp.step()

        if batch_idx % 500 == 0:
            print(f'=> batch_idx: {batch_idx}')

    save_checkpoint(Discriminator, opt_Disc_comp, filename="./saved_model/Discriminator.pth.tar")