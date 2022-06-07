from __future__ import print_function
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from six.moves import urllib
import Config
from model import SpatialTransformerNetwork

plt.ion()   # interactive mode

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# STN Test dataset
transform = transforms.Compose([
    transforms.Resize(Config.image_size),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(Config.channels)], [0.5 for _ in range(Config.channels)]
    ),
])
test_dataset = datasets.ImageFolder(root="a", transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=True)

STN = SpatialTransformerNetwork().to(Config.device)


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(Config.device)

        input_tensor = data.cpu()
        transformed_input_tensor = STN.stn.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')


# Main Function
STN = torch.load('./saved_model/STN_Model.pt')
STN = STN.to(Config.device)

# Visualize the STN transformation on some input batch
visualize_stn()

plt.ioff()
plt.show()