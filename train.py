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

parser = argparse.ArgumentParser(description='Memory replay')

parser.add_argument('--config',  type=str, default='./configs/mnist_mr.yaml')
parser.add_argument('--run',  type=int, default=0)
parser.add_argument('--mem_size',  type=int, default=None)
parser.add_argument('overrides', nargs='*', help="Any key=svalue arguments to override config values "
                                                "(use dots for.nested=overrides)")
flags =  parser.parse_args()
overrides = OmegaConf.from_cli(flags.overrides)
cfg = OmegaConf.load(flags.config)
args = OmegaConf.merge(cfg, overrides)

run = flags.run

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
seed = args.experiment.seed + run * 10
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

if flags.mem_size:
    mem_size = flags.mem_size
else:
    mem_size = args.training.mem_size
    
scenario = args.experiment.scenario

if args.training.surrogate == 'fast_sigmoid':
    spike_grad = surrogate.fast_sigmoid(slope=25)
else:
    spike_grad = surrogate.atan(alpha=2.0)


#--------------------------------------------------
# Load  dataset
#--------------------------------------------------
img_size = args.experiment.img_size
dataset_name = args.experiment.dataset

if dataset_name == 'mnist':
    task_labels = [[0,1],[2,3],[4,5],[6,7],[8,9]]
    num_tasks=len(task_labels)

train_dataset, val_dataset, test_dataset = task_construction(task_labels, args.experiment.dataset, img_size, seed)

#--------------------------------------------------
# Instantiate the SNN model
#--------------------------------------------------
if scenario == 'cil':
    net = CilModel(beta, spike_grad, batch_size, device).to(device)
elif scenario == 'tf':
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
print('********** SNN training and evaluation **********')

mems_x = torch.zeros(num_tasks, mem_size, batch_size, 1, img_size, img_size)
mems_xai = torch.zeros(num_tasks, mem_size, batch_size, 1, img_size, img_size)
mems_y = torch.zeros(num_tasks, mem_size, batch_size, dtype=torch.long)

mem_optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999)) # lr 1e-5 cil
optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

for task_idx in range(0, num_tasks):
    print(f'\nTASK {task_idx}: classes {task_labels[task_idx]}')
    
    # get current task data
    train_loader = get_task_load_train(train_dataset[task_idx], batch_size)
    val_loader = get_task_load_test(val_dataset[task_idx], batch_size)
    
    if scenario == 'cil':
        net.prev_model_adaptation(len(task_labels[task_idx]))
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    best_loss = 10000
    loss_hist = []
    test_acc_hist = []
    curr_mem_size = 0

    # Outer training loop
    for epoch in range(num_epochs):
        print(f'\nEPOCH {epoch}')
        
        # Retraining old tasks
        if args.experiment.replay_raw and task_idx > 0:
            print(f'\nRe-training old tasks...')
            for old_task in range(0, task_idx):
                for mem_idx, mem_batch in enumerate(mems_x[old_task]):
                    data = mem_batch.to(device)
                    targets = mems_y[old_task,mem_idx].long().to(device)
                    
                    if scenario == 'cil':
                        targets = torch.where(targets == task_labels[task_idx][0], 0, 1)

                    # Forward pass
                    net.train()
                    
                    if scenario == 'cil':
                        spk_rec, _, _ = forward_pass(net, num_steps, data, old_task)
                    else:
                        spk_rec, _, _ = forward_pass(net, num_steps, data)

                    # Initialize the loss & sum over time
                    loss_val = loss_fn(spk_rec, targets)

                    # Gradient calculation + weight update
                    mem_optimizer.zero_grad()
                    loss_val.backward()
                    mem_optimizer.step()
                    
        counter = 0

        # Training loop
        for data, targets in iter(train_loader):
            data = spikegen.rate_conv(data)
            
            storing_check = False
            if args.experiment.replay_raw:
                # Storing evidence for current task
                if random.random() < 0.33 and curr_mem_size < mem_size:
                    mems_x[task_idx, curr_mem_size] = data
                    mems_y[task_idx, curr_mem_size] = targets
                    curr_mem_size += 1
                    storing_check = True

            data = data.to(device)
            
            if scenario == 'cil':
                targets = torch.where(targets == task_labels[task_idx][0], 0, 1)
                
            targets = targets.to(device)

            # forward pass
            net.train()
            if scenario == 'cil':
                spk_rec, _, saved_forward = forward_pass(net, num_steps, data, task_idx)
            else:
                spk_rec, _, saved_forward = forward_pass(net, num_steps, data)
                
            #if args.experiment.replay_xai and storing_check:
                

            # initialize the loss & sum over time
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Test set
            if counter % 100 == 0:
                with torch.no_grad():
                    net.eval()

                    # Test set forward pass
                    if scenario == 'cil':
                        test_acc = batch_accuracy(val_loader, net, num_steps, device, task_idx, task_labels)
                    else:
                        test_acc = batch_accuracy(val_loader, net, num_steps, device)
                    print(f"Iteration {counter}, Test Acc Task {task_idx}: {test_acc * 100:.2f}%")
                    test_acc_hist.append(test_acc.item())

                    if loss_val < best_loss:
                        model_dict = {
                                'task_idx': task_idx,
                                'state_dict': net.state_dict(),
                                'accuracy': test_acc.item()}

                        torch.save(model_dict, log_dir+f'/run_{run}_task_{task_idx}_scenario_{scenario}_lr_{lr}_memsize_{mem_size}.pth.tar')
                        print('Saving the model...\n')

            counter += 1
                 
    # Normalizing the weights
    if scenario == 'cil':
        net.after_model_adaptation(len(task_labels[task_idx]), task_idx)

sys.exit(0)