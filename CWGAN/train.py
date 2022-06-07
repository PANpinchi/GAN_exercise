"""
Training of DCGAN network on MNIST dataset with Discriminator
and Generator imported from model.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
from utils import gradient_penalty

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 5e-5
batch_size = 64
image_size = 64
channels_img = 3
num_classes = 10
gen_embedding = 100
z_dim = 100
num_epochs = 1000
features_d = 64
features_g = 64
critic_iterations = 5
lambda_gp = 10

transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
    ),
])

# dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
dataset = datasets.ImageFolder(root="logo", transform=transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
gen = Generator(z_dim, channels_img, features_g, num_classes, image_size, gen_embedding).to(device)
critic = Discriminator(channels_img, features_d, num_classes, image_size).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
critic.train()

for epoch in range(num_epochs):
    for batch_idx, (real, labels) in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        labels = labels.to(device)

        for _ in range(critic_iterations):
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            try:
                fake = gen(noise, labels)
            except:
                continue
            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)
            try:
                gp = gradient_penalty(critic, labels, real, fake, device=device)
            except:
                continue
            loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp)
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: min -E[critic(gen_fake)]
        try:
            output = critic(fake, labels).reshape(-1)
        except:
            continue
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                              Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise, labels)
                # take out (up to) 32 examples
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image(
                    "Fake", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Real", img_grid_real, global_step=step
                )
                step += 1

# Use Anaconda Prompt open tensorboard
# command : tensorboard --samples_per_plugin images=49 --logdir logs
# --samples_per_plugin images=[num_epochs]
# --logdir [file_name]