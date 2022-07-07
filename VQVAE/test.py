from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import make_grid


def draw_sample_image(x, postfix):
    plt.figure(figsize=(16, 16))
    plt.axis("off")
    plt.title("Visualization of {}".format(postfix))
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))


vqvae = torch.load('./vqvae.pth')
vqvae.eval()

vqvae_data_dir = './vqvae_datasets'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(384),
        transforms.ColorJitter(contrast=0.5, saturation=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(vqvae_data_dir, x),
                  data_transforms[x])
                  for x in ['train', 'val']}

batch_size = 8
dataloaders = {x: torch.utils.data.DataLoader(\
              image_datasets[x], batch_size=batch_size,
              shuffle=True, num_workers=2)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    for batch_idx, (x, _) in enumerate(tqdm(dataloaders['train'])):
        x = x.to(device)
        x_hat, commitment_loss, codebook_loss, perplexity = vqvae(x)

        z = vqvae.encoder(x)
        z_quantized, commitment_loss, codebook_loss, perplexity = vqvae.codebook(z)

        print("perplexity: ", perplexity.item(),
              "commit_loss: ", commitment_loss.item(),
              "  codebook loss: ", codebook_loss.item())
        break

draw_sample_image(x[:], "Ground-truth images")
draw_sample_image(x_hat[:], "Reconstructed images")