"""
Training of Compositional GAN on Custom dataset with Discriminator
and Generator imported from model.py
"""

import torch
from torch import index_select
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from model import Discriminator_STN, SpatialTransformerNetwork
from model import CompositionGenerator, DecompositionGenerator, initialize_weights
from model import HierarchyGenerator, LogoStyleGenerator
from utils import save_checkpoint, load_checkpoint
import itertools
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
Comp_Gen = CompositionGenerator(Config.channels).to(Config.device)
Hierarchy_Gen = HierarchyGenerator().to(Config.device)
Style_Gen = LogoStyleGenerator().to(Config.device)
Decomp_Gen = DecompositionGenerator(Config.channels).to(Config.device)
Comp_STN = SpatialTransformerNetwork().to(Config.device)
Decomp_STN = SpatialTransformerNetwork().to(Config.device)

opt_Disc = optim.Adam(Discriminator.parameters(), lr=Config.lr, betas=(0.5, 0.999))
opt_Gen = optim.Adam(itertools.chain(Comp_Gen.parameters(),
                                     Decomp_Gen.parameters(),
                                     Hierarchy_Gen.parameters(),
                                     Style_Gen.parameters()), lr=Config.lr, betas=(0.5, 0.999))
opt_STN = optim.SGD(itertools.chain(Comp_STN.parameters(), Decomp_STN.parameters()), lr=Config.stn_lr)

# initialize or load weights
if Config.load_model:
    load_checkpoint("./saved_model/Comp_Gen.pth.tar", Comp_Gen, opt_Gen, Config.lr,)
    load_checkpoint("./saved_model/Decomp_Gen.pth.tar", Decomp_Gen, opt_Gen, Config.lr)
    load_checkpoint("./saved_model/Discriminator.pth.tar", Discriminator, opt_Disc, Config.lr)
    load_checkpoint("./saved_model/Comp_STN.pth.tar", Comp_STN, opt_STN, Config.lr)
    load_checkpoint("./saved_model/Decomp_STN.pth.tar", Decomp_STN, opt_STN, Config.lr)
else:
    initialize_weights(Discriminator)
    initialize_weights(Comp_Gen)
    initialize_weights(Hierarchy_Gen)
    initialize_weights(Style_Gen)
    initialize_weights(Decomp_Gen)
    initialize_weights(Comp_STN)
    initialize_weights(Decomp_STN)

L1 = nn.L1Loss()
mse = nn.MSELoss()

G_scaler = torch.cuda.amp.GradScaler()
D_scaler = torch.cuda.amp.GradScaler()
STN_scaler = torch.cuda.amp.GradScaler()

Discriminator.train()
Comp_Gen.train()
Hierarchy_Gen.train()
Style_Gen.train()
Decomp_Gen.train()
Comp_STN.train()
Decomp_STN.train()

steps = 0
img_num = 0
pre_fake_score = 0
best_fake_score = 0
real_score_arr = []
fake_score_arr = []
batch_idx_arr = []
print('Start Training Compositional Model')
for epoch in range(1, Config.num_epochs + 1):
    for (batch_idx, (real_comp, _)) in enumerate(comp_loader):
        for (_, (a, _)), (_, (b, _)), (_, (c, _)) in zip(enumerate(a_loader), enumerate(b_loader), enumerate(c_loader)):
            torch.cuda.empty_cache()

            a = a.to(Config.device)
            b = b.to(Config.device)
            c = c.to(Config.device)
            real_comp = real_comp.to(Config.device)

            if Config.change_style:
                a_b_c = torch.cat([a, b, c], 1).to(Config.device)
                xs = Style_Gen(a_b_c)
                ones = torch.ones_like(a)
                zero = torch.zeros_like(a)
                a_mask = torch.where(a < 0, ones, zero)
                b_mask = torch.where(b < 0, ones, zero)
                c_mask = torch.where(c < 0, ones, zero)
                a_out_mask = torch.where(a < 0, zero, ones)
                b_out_mask = torch.where(b < 0, zero, ones)
                c_out_mask = torch.where(c < 0, zero, ones)
                a = xs[0][0] * a_mask + a_out_mask
                b = xs[0][1] * b_mask + b_out_mask
                c = xs[0][2] * c_mask + c_out_mask

            a_b_c = torch.cat([a, b, c], 1).to(Config.device)
            a_T, b_T, c_T = Comp_STN(a_b_c)
            a_T_b_T_c_T = torch.cat([a_T, b_T, c_T], 1).to(Config.device)
            a_rT, b_rT, c_rT = Decomp_STN(a_T_b_T_c_T)

            # # # # # # # # # # # # # # # # # # #
            # Train Compositional Discriminator #
            # # # # # # # # # # # # # # # # # # #

            torch.cuda.empty_cache()
            aab = torch.cat([a_T, a_T, b_T], 1).to(Config.device)
            bbc = torch.cat([b_T, b_T, c_T], 1).to(Config.device)
            acc = torch.cat([a_T, c_T, c_T], 1).to(Config.device)
            fake_ab = Comp_Gen(aab)
            fake_bc = Comp_Gen(bbc)
            fake_ac = Comp_Gen(acc)

            # Only Train Comp STN Disc
            # Decomp Disc Using L1 Loss
            Disc_STN_real_comp = Discriminator(real_comp)

            Disc_STN_fake_ab = Discriminator(fake_ab)
            Disc_STN_fake_bc = Discriminator(fake_bc)
            Disc_STN_fake_ac = Discriminator(fake_ac)

            Disc_STN_a_T = Discriminator(a_T)
            Disc_STN_b_T = Discriminator(b_T)
            Disc_STN_c_T = Discriminator(c_T)

            Disc_STN_ones_comp = Discriminator(torch.ones_like(real_comp))
            Disc_STN_any_comp = Discriminator(torch.ones_like(real_comp) * torch.mean(xs))
            Disc_STN_minus_ones_comp = Discriminator(-torch.ones_like(real_comp))

            # Calculate Loss
            Disc_STN_real_comp_loss = mse(Disc_STN_real_comp, torch.ones_like(Disc_STN_real_comp))

            Disc_STN_fake_ab_loss = mse(Disc_STN_fake_ab, torch.ones_like(Disc_STN_fake_ab) * 2 / 3)
            Disc_STN_fake_bc_loss = mse(Disc_STN_fake_bc, torch.ones_like(Disc_STN_fake_bc) * 2 / 3)
            Disc_STN_fake_ac_loss = mse(Disc_STN_fake_ac, torch.ones_like(Disc_STN_fake_ac) * 2 / 3)

            Disc_STN_a_T_loss = mse(Disc_STN_a_T, torch.ones_like(Disc_STN_a_T) / 3)
            Disc_STN_b_T_loss = mse(Disc_STN_b_T, torch.ones_like(Disc_STN_b_T) / 3)
            Disc_STN_c_T_loss = mse(Disc_STN_c_T, torch.ones_like(Disc_STN_c_T) / 3)

            Disc_STN_ones_comp_loss = mse(Disc_STN_ones_comp, torch.zeros_like(Disc_STN_ones_comp))
            Disc_STN_any_comp_loss = mse(Disc_STN_any_comp, torch.zeros_like(Disc_STN_any_comp))
            Disc_STN_minus_ones_comp_loss = mse(Disc_STN_minus_ones_comp, torch.zeros_like(Disc_STN_minus_ones_comp))

            Disc_Comp_STN_loss = (
                Disc_STN_real_comp_loss +
                Disc_STN_fake_ab_loss +
                Disc_STN_fake_bc_loss +
                Disc_STN_fake_ac_loss +
                Disc_STN_a_T_loss +
                Disc_STN_b_T_loss +
                Disc_STN_c_T_loss +
                Disc_STN_ones_comp_loss +
                Disc_STN_any_comp_loss +
                Disc_STN_minus_ones_comp_loss
            )

            Discriminator.zero_grad()
            Disc_Comp_STN_loss.backward(retain_graph=True)
            opt_Disc.step()

            # # # # # # # # # # # # # # # # # # #
            # Train Spatial Transformer Network #
            # # # # # # # # # # # # # # # # # # #

            torch.cuda.empty_cache()

            # Generate Fake Comp
            fake_comp = Comp_Gen(a_T_b_T_c_T)

            # Disc Comp
            D_fake_comp = Discriminator(fake_comp)

            # Comp loss
            STN_fake_comp_loss = mse(D_fake_comp, torch.ones_like(D_fake_comp))

            # Decomp loss
            L1_STN_a_rT_loss = L1(a, a_rT)
            L1_STN_b_rT_loss = L1(b, b_rT)
            L1_STN_c_rT_loss = L1(c, c_rT)

            STN_loss = (
                STN_fake_comp_loss +
                L1_STN_a_rT_loss +
                L1_STN_b_rT_loss +
                L1_STN_c_rT_loss
            )

            # size loss
            size_a = torch.sum(torch.abs(a - torch.ones_like(a))) / (128 * 128)
            size_b = torch.sum(torch.abs(b - torch.ones_like(b))) / (128 * 128)
            size_c = torch.sum(torch.abs(c - torch.ones_like(c))) / (128 * 128)
            size_a_T = torch.sum(torch.abs(a_T - torch.ones_like(a_T))) / (128 * 128)
            size_b_T = torch.sum(torch.abs(b_T - torch.ones_like(b_T))) / (128 * 128)
            size_c_T = torch.sum(torch.abs(c_T - torch.ones_like(c_T))) / (128 * 128)

            STN_size_loss = (
                L1(size_a, size_a_T) +
                L1(size_b, size_b_T) +
                L1(size_c, size_c_T) +
                L1(size_a_T, size_b_T) +
                L1(size_b_T, size_c_T) +
                L1(size_c_T, size_a_T)
            )

            STN_loss = (
                    Config.lambda_STN * STN_loss +
                    Config.lambda_Size * STN_size_loss
            )

            opt_STN.zero_grad()
            STN_scaler.scale(STN_loss).backward(retain_graph=True)
            STN_scaler.step(opt_STN)
            STN_scaler.update()

            # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Train Compositional and Decompositional Generators  #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            torch.cuda.empty_cache()

            a_b_c = torch.cat([a, b, c], 1).to(Config.device)
            a_T, b_T, c_T = Comp_STN(a_b_c)
            real_decomp = torch.cat([a_T, b_T, c_T], 1).to(Config.device)

            # Comp
            fake_comp = Comp_Gen(real_decomp)  # Generate Fake Compositional Image From Real Decompositional Image
            Disc_comp_fake = Discriminator(fake_comp)  # Discriminate Fake Compositional Image

            # Color loss
            real_color = torch.mean(real_comp)
            fake_color = torch.mean(fake_comp)
            color_loss = (
                    L1(xs, torch.ones_like(xs) * real_color) +
                    L1(fake_color, real_color)
            )

            # Obj loss
            obj_loss = (
                -L1(xs[0][0], xs[0][1]) +
                -L1(xs[0][1], xs[0][2]) +
                -L1(xs[0][2], xs[0][0])
            )

            # Decomp
            fake_decomp = Decomp_Gen(fake_comp)  # Generate Fake Decompositional Image From Real Compositional Image
            fake_a_T = index_select(fake_decomp, 1, Config.ind1)
            fake_b_T = index_select(fake_decomp, 1, Config.ind2)
            fake_c_T = index_select(fake_decomp, 1, Config.ind3)

            # GAN loss
            Gen_loss = mse(Disc_comp_fake, torch.ones_like(Disc_comp_fake))

            # Cycle loss
            cycle_comp = Comp_Gen(fake_decomp)
            cycle_decomp = Decomp_Gen(cycle_comp)
            cycle_decomp_A = index_select(cycle_decomp, 1, Config.ind1)
            cycle_decomp_B = index_select(cycle_decomp, 1, Config.ind2)
            cycle_decomp_C = index_select(cycle_decomp, 1, Config.ind3)

            cycle_comp_loss = L1(cycle_comp, fake_comp)
            cycle_decomp_A_loss = L1(cycle_decomp_A, a_T)
            cycle_decomp_B_loss = L1(cycle_decomp_B, b_T)
            cycle_decomp_C_loss = L1(cycle_decomp_C, c_T)

            # L1 loss
            L1_G_A_T = L1(fake_a_T, a_T)
            L1_G_B_T = L1(fake_b_T, b_T)
            L1_G_C_T = L1(fake_c_T, c_T)

            # Comp loss
            ones = torch.ones_like(fake_comp)
            zero = torch.zeros_like(fake_comp)
            a_T_mask = torch.where(a_T < 0.9, ones, zero)
            b_T_mask = torch.where(b_T < 0.9, ones, zero)
            c_T_mask = torch.where(c_T < 0.9, ones, zero)
            out_mask = torch.where(a_T_mask + b_T_mask + c_T_mask == 0, ones, zero)
            a_T_mask_loss = L1(fake_comp * a_T_mask, a_T * a_T_mask)
            b_T_mask_loss = L1(fake_comp * b_T_mask, b_T * b_T_mask)
            c_T_mask_loss = L1(fake_comp * c_T_mask, c_T * c_T_mask)
            out_mask_loss = L1(fake_comp * out_mask, torch.ones_like(fake_comp) * out_mask)

            # Calculate Object Weight
            hierarchy = Hierarchy_Gen(real_decomp)
            h0 = hierarchy[0][0].detach() if hierarchy[0][0].detach() != 0 else 0.000001
            h1 = hierarchy[0][1].detach() if hierarchy[0][1].detach() != 0 else 0.000001
            h2 = hierarchy[0][2].detach() if hierarchy[0][2].detach() != 0 else 0.000001
            _, ind = hierarchy.sort()
            if ind[0][0] == 0:
                if hierarchy[0][0] != 0:
                    a_weight = hierarchy[0][0] * 1 / h0
                else:
                    a_weight = 2
            elif ind[0][0] == 1:
                if hierarchy[0][1] != 0:
                    b_weight = hierarchy[0][1] * 1 / h1
                else:
                    b_weight = 1
            else:
                if hierarchy[0][2] != 0:
                    c_weight = hierarchy[0][2] * 1 / h2
                else:
                    c_weight = 1

            if ind[0][1] == 0:
                if hierarchy[0][0] != 0:
                    a_weight = hierarchy[0][0] * 2 / h0
                else:
                    a_weight = 2
            elif ind[0][1] == 1:
                if hierarchy[0][1] != 0:
                    b_weight = hierarchy[0][1] * 2 / h1
                else:
                    b_weight = 2
            else:
                if hierarchy[0][2] != 0:
                    c_weight = hierarchy[0][2] * 2 / h2
                else:
                    c_weight = 2

            if ind[0][2] == 0:
                if hierarchy[0][0] != 0:
                    a_weight = hierarchy[0][0] * 3 / h0
                else:
                    a_weight = 3
            elif ind[0][2] == 1:
                if hierarchy[0][1] != 0:
                    b_weight = hierarchy[0][1] * 3 / h1
                else:
                    b_weight = 3
            else:
                if hierarchy[0][2] != 0:
                    c_weight = hierarchy[0][2] * 3 / h2
                else:
                    c_weight = 3

            Comp_loss = (
                    a_weight * 30 * a_T_mask_loss +
                    b_weight * 30 * b_T_mask_loss +
                    c_weight * 30 * c_T_mask_loss +
                    10 * out_mask_loss
            )

            # add all together
            G_loss = (
                # GAN loss
                Config.lambda_GAN * (
                    Gen_loss
                ) +
                # Comp loss
                Config.lambda_Comp * (
                    Comp_loss
                ) +
                # L1 loss
                Config.lambda_L1 * (
                    L1_G_A_T +
                    L1_G_B_T +
                    L1_G_C_T
                ) +
                # Cycle loss
                Config.lambda_Cycle * (
                    cycle_comp_loss +
                    cycle_decomp_A_loss +
                    cycle_decomp_B_loss +
                    cycle_decomp_C_loss
                ) +
                Config.lambda_Color * color_loss +
                Config.lambda_Obj * obj_loss
            )

            # Update Generator Weight
            opt_Gen.zero_grad()
            G_scaler.scale(G_loss).backward()
            G_scaler.step(opt_Gen)
            G_scaler.update()

            if batch_idx % 10 == 0 or batch_idx < 20:
                with torch.no_grad():
                    real_score = Discriminator(real_comp)  # Discriminate Fake Compositional Image
                    fake_score = Discriminator(fake_comp)  # Discriminate Fake Compositional Image
                    real_score = torch.mean(real_score)
                    fake_score = torch.mean(fake_score)
                    fake_a = index_select(fake_decomp, 1, Config.ind1)
                    fake_b = index_select(fake_decomp, 1, Config.ind2)
                    fake_c = index_select(fake_decomp, 1, Config.ind3)

                if epoch == 1 and batch_idx == 0:
                    real_score_arr.append(real_score)
                    fake_score_arr.append(fake_score)
                    batch_idx_arr.append((epoch - 1) * 5000 + batch_idx)

                    plt.figure(0)
                    plt.plot(np.array(batch_idx_arr), np.array(real_score_arr), 'r-', label="real score", linewidth=0.8)
                    plt.plot(np.array(batch_idx_arr), np.array(fake_score_arr), 'b-', label="fake score", linewidth=0.8)
                    plt.xlabel('Epoch')
                    plt.ylabel('Score')
                    plt.legend()
                    plt.savefig("Score.png")
                elif batch_idx % 10 == 0:
                    real_score_arr.append(real_score)
                    fake_score_arr.append(fake_score)
                    batch_idx_arr.append((epoch - 1) * 5000 + batch_idx)

                    plt.plot(np.array(batch_idx_arr), np.array(real_score_arr), 'r-', linewidth=0.8)
                    plt.plot(np.array(batch_idx_arr), np.array(fake_score_arr), 'b-', linewidth=0.8)
                    plt.xlabel('Epoch')
                    plt.ylabel('Score')
                    plt.legend()
                    plt.savefig("Score.png")

                # Save Image to 'saved_images' Folder
                save_image(real_comp * 0.5 + 0.5, f"saved_images/real_comp/comp_{img_num}.png")
                save_image(fake_ab * 0.5 + 0.5, f"saved_images/real_a_T/decomp_a_{img_num}.png")
                save_image(fake_bc * 0.5 + 0.5, f"saved_images/real_b_T/decomp_b_{img_num}.png")
                save_image(fake_ac * 0.5 + 0.5, f"saved_images/real_c_T/decomp_c_{img_num}.png")
                save_image(fake_comp * 0.5 + 0.5, f"saved_images/fake_comp/comp_{img_num}.png")
                save_image(fake_a * 0.5 + 0.5, f"saved_images/fake_a_T/decomp_a_{img_num}.png")
                save_image(fake_b * 0.5 + 0.5, f"saved_images/fake_b_T/decomp_b_{img_num}.png")
                save_image(fake_c * 0.5 + 0.5, f"saved_images/fake_c_T/decomp_c_{img_num}.png")
                save_image(a_rT * 0.5 + 0.5, f"saved_images/fake_a_rT/decomp_a_{img_num}.png")
                save_image(b_rT * 0.5 + 0.5, f"saved_images/fake_b_rT/decomp_b_{img_num}.png")
                save_image(c_rT * 0.5 + 0.5, f"saved_images/fake_c_rT/decomp_c_{img_num}.png")

                # Print Losses
                print(
                    f'Epoch [{epoch:2d}/{Config.num_epochs}] '
                    f'Batch [{batch_idx:4d}/{len(comp_loader)}] '
                    f'Image: {img_num: 3d}  '
                    f'Real Score: {real_score:6.4f}, '
                    f'Fake Score: {fake_score:6.4f}, '
                    f'Loss D: {Disc_Comp_STN_loss:6.4f}, '
                    f'Loss G: {G_loss:9.4f}, '
                    f'Loss STN: {STN_loss:6.4f}, '
                    f'size_a_T: {size_a_T:6.4f}, '
                    f'size_b_T: {size_b_T:6.4f}, '
                    f'size_c_T: {size_c_T:6.4f}, '
                    f'a_weight: {a_weight: .0f}, '
                    f'b_weight: {b_weight: .0f}, '
                    f'c_weight: {c_weight: .0f}, '
                    f'a_hierarchy: {hierarchy[0][0]: 6.4f}, '
                    f'b_hierarchy: {hierarchy[0][1]: 6.4f}, '
                    f'c_hierarchy: {hierarchy[0][2]: 6.4f}, '
                    f'xs[0][0]: {xs[0][0]: 6.4f}, '
                    f'xs[0][1]: {xs[0][1]: 6.4f}, '
                    f'xs[0][2]: {xs[0][2]: 6.4f}  '
                )

                if fake_score > best_fake_score:
                    best_fake_score = fake_score
                    best_img_num = img_num
                    best_img = fake_comp

                if img_num > 50 and pre_fake_score > 0.5 and fake_score < 0.5:
                    save_image(best_img * 0.5 + 0.5, f"logo/output.png")
                    print(f'=> Best Image: {best_img_num}')
                    print('=> LOGO Generate Completed')
                    quit()
                else:
                    pre_fake_score = fake_score

                if fake_score >= 0.95:
                    save_image(best_img * 0.5 + 0.5, f"logo/output.png")
                    print(f'=> Best Image: {best_img_num}')
                    print('=> LOGO Generate Completed')
                    quit()
                else:
                    img_num += 1

    # Save Models
    save_checkpoint(Comp_Gen, opt_Gen, filename="./saved_model/Comp_Gen.pth.tar")
    save_checkpoint(Decomp_Gen, opt_Gen, filename="./saved_model/Decomp_Gen.pth.tar")
    save_checkpoint(Discriminator, opt_Disc, filename="./saved_model/Discriminator.pth.tar")
    save_checkpoint(Comp_STN, opt_STN, filename="./saved_model/Comp_STN.pth.tar")
    save_checkpoint(Decomp_STN, opt_STN, filename="./saved_model/Decomp_STN.pth.tar")
