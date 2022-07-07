import torch.optim as optim
from torchvision import datasets, transforms
import os
import torch
import torch.nn as nn
from model import Model, Encoder, VQEmbeddingEMA, Decoder

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

input_dim = 3
hidden_dim = 128
n_embeddings = 768
output_dim = 3

encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
codebook = VQEmbeddingEMA(n_embeddings=n_embeddings, embedding_dim=hidden_dim)
decoder = Decoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim)

vqvae = Model(Encoder=encoder, Codebook=codebook, Decoder=decoder).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(vqvae.parameters(), lr=0.0001)

print("Start training VQ-VAE...")
vqvae.train()
epochs = 20

for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(dataloaders['train']):
        x = x.to(device)

        optimizer.zero_grad()

        x_hat, commitment_loss, codebook_loss, perplexity = vqvae(x)
        recon_loss = criterion(x_hat, x)

        loss = recon_loss + commitment_loss + codebook_loss

        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print("epoch:", epoch + 1, "  step:", batch_idx + 1, "  recon_loss:", recon_loss.item(), "  perplexity: ",
                  perplexity.item(),
                  "\n\t\tcommit_loss: ", commitment_loss.item(), "  codebook loss: ", codebook_loss.item(),
                  "  total_loss: ", loss.item())

torch.save(vqvae, 'vqvae.pth')
print("Finish!!")