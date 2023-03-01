import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import scipy
import pickle
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.losses import kl_divergence_from_nn
from blitz.utils import variational_estimator
from scipy.stats import bernoulli
import numpy as np
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor
batch_size = 100
class BayesianCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BayesianConv2d(1, 6, (5, 5))
        self.conv2 = BayesianConv2d(6, 16, (5, 5))
        self.fc1 = BayesianLinear(256, 120)
        self.fc2 = BayesianLinear(120, 84)
        self.fc3 = BayesianLinear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def evaluate_hw1(model_name):
    # load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = pickle.load(open(model_name, 'rb'))
    test_dataset = dsets.MNIST(root='./data', train=False, transform=ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    num_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    kl_full_model=(kl_divergence_from_nn(model).item())
    kl_divided_by_params = kl_full_model / num_of_parameters
    print("KL divergence of the model is: ", kl_full_model)
    print("KL divergence divided by the number of parameters is: ", kl_divided_by_params)
#2.1
evaluate_hw1("q2_full_model.pkl")
#2.2
evaluate_hw1("q2_first_200.pkl")
#2.3
evaluate_hw1("q2_first_200_only_2_8.pkl")
#2.4
evaluate_hw1("q2_only_2_8.pkl")
#2.5
evaluate_hw1("q2_random.pkl")

