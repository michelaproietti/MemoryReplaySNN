#--------------------------------------------------
# Imports
#--------------------------------------------------
import os
import sys
import torch
import random
import argparse
import itertools
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import spikegen
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

from model import CilModel, Net
from utils import forward_pass, batch_accuracy, compute_CAM
from data import task_construction, get_task_load_train, get_task_load_test

#--------------------------------------------------
# Parse input arguments
#--------------------------------------------------
args = OmegaConf.load('./configs/mnist_mr.yaml')

#--------------------------------------------------
# Set device
#--------------------------------------------------

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#--------------------------------------------------
# Initialize tensorboard setting
#--------------------------------------------------
log_dir = args.paths.checkpoints
if os.path.isdir(log_dir) is not True:
    os.mkdir(log_dir)

#--------------------------------------------------
# Initialize seed
#--------------------------------------------------
seed = args.experiment.seed + 10*2
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

#--------------------------------------------------
# SNN configuration parameters
#--------------------------------------------------
# SNN learning and evaluation parameters
batch_size      = args.training.batch_size
num_epochs      = args.training.num_epochs
num_steps       = args.training.num_steps
lr   = args.training.lr
beta = args.training.beta

if args.training.surrogate == 'fast_sigmoid':
    spike_grad = surrogate.fast_sigmoid(slope=25)
else:
    spike_grad = surrogate.atan(alpha=2.0)


#--------------------------------------------------
# Load  dataset
#--------------------------------------------------
img_size = args.experiment.img_size
dataset_name = args.experiment.dataset

if dataset_name == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif dataset_name == 'mnist':
    transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
        
    full_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [int(0.9*len(full_dataset)), int(0.1*len(full_dataset))])
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

#--------------------------------------------------
# Instantiate the SNN model
#--------------------------------------------------
net = Net(beta, spike_grad, batch_size).to(device)

# Configure the loss function and optimizer
loss_fn = SF.ce_count_loss()

# Print the SNN model, optimizer, and simulation parameters
print('********** SNN simulation parameters **********')
print('Simulation # time-step : {}'.format(num_steps))

print('********** SNN learning parameters **********')
print('Backprop optimizer     : Adam')
print('Batch size             : {}'.format(batch_size))
print('Number of epochs       : {}'.format(num_epochs))
print('Learning rate          : {}'.format(lr))

#--------------------------------------------------
# Train the SNN using surrogate gradients
#--------------------------------------------------
print('********** SNN training ************')

optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
best_loss = 1000000

for epoch in range(num_epochs):
    print(f'\nEPOCH {epoch}')

    counter = 0

    # Training loop
    for data, targets in iter(train_loader):
        data = spikegen.rate_conv(data)

        data = data.to(device)                
        targets = targets.to(device)

        # Forward pass
        net.train()
        spk_rec, _, saved_forward = forward_pass(net, num_steps, data)            

        # Initialize the loss & sum over time
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Test set
        if counter % 100 == 0:
            with torch.no_grad():
                net.eval()

                # Test set forward pass
                test_acc = batch_accuracy(val_loader, net, num_steps, device)
                print(f"Iteration {counter}, Test Accuracy = {test_acc * 100:.2f}%")

                if loss_val < best_loss:
                    model_dict = {
                            'epoch': epoch,
                            'state_dict': net.state_dict(),
                            'accuracy': test_acc.item()}

                    torch.save(model_dict, log_dir+f'/joint_training_lr_{lr}.pth.tar')
                    print('Saving the model...\n')

        counter += 1
        
#--------------------------------------------------
# Evaluate the SNN
#--------------------------------------------------
print('********** SNN evaluation ************')
net = Net(beta, spike_grad, batch_size).to(device)

model_dict = torch.load(log_dir+f'/joint_training_lr_{lr}.pth.tar')
net.load_state_dict(model_dict['state_dict'])

with torch.no_grad():
    net.eval()

    test_acc = batch_accuracy(test_loader, net, num_steps, device)
    
    print(f"Test Accuracy = {test_acc * 100:.2f}%") 

sys.exit(0)