import torch
from torch import LongTensor
from torch.autograd import Variable

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False

train_stn = False
train_compsition = True
change_style = True

lr = 5e-5
stn_lr = 1e-3
batch_size = 1
image_size = 128
channels = 1

num_epochs = 99
disc_iterations = 5

inp1 = Variable(LongTensor(range(0, 2))).to(device)
inp2 = Variable(LongTensor(range(2, 4))).to(device)
inp3 = Variable(LongTensor(range(4, 6))).to(device)
ind1 = torch.tensor([0]).to(device)
ind2 = torch.tensor([1]).to(device)
ind3 = torch.tensor([2]).to(device)

lambda_L1 = 100
lambda_Comp = 100
lambda_Color = 1000
lambda_Cycle = 10
lambda_Size = 1
lambda_Obj = 100
lambda_GAN = 10
lambda_STN = 10
