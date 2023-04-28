#--------------------------------------------------
# Imports
#--------------------------------------------------
import os
import sys
import json
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import itertools
import argparse
from omegaconf import OmegaConf

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
parser.add_argument('--run',  type=str, default=0)
parser.add_argument('--mem_size',  type=str, default=None)
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
    
stats_dir = args.paths.stats_dir
if os.path.isdir(stats_dir) is not True:
    os.mkdir(stats_dir)
    
cam_dir = args.paths.cam_dir
if os.path.isdir(cam_dir) is not True:
    os.mkdir(cam_dir) 

#--------------------------------------------------
# Initialize seed
#--------------------------------------------------
seed = args.experiment.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

#--------------------------------------------------
# SNN configuration parameters
#--------------------------------------------------
# SNN learning and evaluation parameters
batch_size      = 1
num_epochs      = args.training.num_epochs
num_steps       = args.training.num_steps
lr   = args.training.lr
beta = args.training.beta

if flags.mem_size:
    mem_size = flags.mem_size
else:
    mem_size = args.training.mem_size
    
scenario = args.experiment.scenario

stats_file = f'run_{run}_scenario_{scenario}_lr_{lr}_memsize_{mem_size}.json'

if args.training.surrogate == 'fast_sigmoid':
    spike_grad = surrogate.fast_sigmoid(slope=25)
else:
    spike_grad = surrogate.atan(alpha=2.0)


#--------------------------------------------------
# Load  dataset
#--------------------------------------------------

if args.experiment.dataset == 'mnist':
    task_labels = [[0,1],[2,3],[4,5],[6,7],[8,9]]
    num_tasks=len(task_labels)
    img_size = 28


train_dataset, val_dataset, test_dataset = task_construction(task_labels, args.experiment.dataset, img_size, seed)

#--------------------------------------------------
# Instantiate the SNN model
#--------------------------------------------------
if args.experiment.scenario == 'cil':
    net = CilModel(beta, spike_grad, batch_size, device).to(device)
else:
    net = Net(beta, spike_grad, batch_size).to(device)

loss_fn = SF.ce_rate_loss()

#--------------------------------------------------
# Test the SNN
#--------------------------------------------------
print('********** SNN testing **********')

accs = torch.zeros(num_tasks, num_tasks)
for i in range(0,num_tasks):
    model_dict = torch.load(log_dir+f'/run_{run}_task_{i}_scenario_{scenario}_lr_{lr}_memsize_{mem_size}.pth.tar')
    if args.experiment.scenario == 'cil':
        net.prev_model_adaptation(len(task_labels[i]))
    net.load_state_dict(model_dict['state_dict'])
    net.eval()

    print(f'Model trained on task {i+1}')

    for j in range(0,i+1):
        if i == j:
            accs[i,j] = model_dict['accuracy']
            print(f'Accuracy on task {j+1}: {model_dict["accuracy"]}')
        else:
            test_loader = get_task_load_test(test_dataset[j], batch_size)
            
            if args.experiment.scenario == 'cil':
                test_acc = batch_accuracy(test_loader, net, num_steps, device)#, j, task_labels, cam=False, cam_dir=cam_dir)
            else:
                test_acc = batch_accuracy(test_loader, net, num_steps, device, cam=False, cam_dir=cam_dir)
                
            accs[i,j] = test_acc.item()

            print(f'Accuracy on task {j+1}: {test_acc.item()}')
    
#--------------------------------------------------
# Save statistics
#--------------------------------------------------

if os.path.isfile(stats_dir+stats_file) is False:
    open(os.path.join(stats_dir+stats_file), 'a').close()

listObj = [] 
with open(stats_dir+stats_file, 'r') as fp:
    if len(fp.readlines()) != 0:
        fp.seek(0)
        listObj = json.load(fp)
        
for task_i in range(0,num_tasks):
    stats_dict = {'Trained task': task_i}
    
    for task_j in range(0, task_i+1):
        stats_dict.update({f'Acc task {task_j}': accs[task_i,task_j].item()})
        
    listObj.append(stats_dict)
    
with open(stats_dir+stats_file, 'w') as json_file:
    json.dump(listObj, json_file, indent=4, separators=(',',': '))

#--------------------------------------------------
# Plot statistics
#--------------------------------------------------

fig = plt.figure(figsize=(12,6))
for i in range(0,num_tasks):
  plt.plot(range(i+1,num_tasks+1), accs[i:,i], label=f'Task {i+1}')

plt.title("MNIST-split")
plt.xlabel("Task")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(f'./training_history/accuracy_history_run_{run}_scenario_{scenario}_memsize{mem_size}.png', dpi=600)

sys.exit(0)