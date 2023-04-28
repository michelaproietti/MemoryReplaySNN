import torch
from snntorch import utils
from snntorch import spikegen
from snntorch import functional as SF

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import skimage
import os

def forward_pass(net, num_steps, data, mem_batch_size=None, task_idx=None):
    mem_rec = []
    spk_rec = []
    conv_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        if task_idx:
            spk_out, mem_out, conv_out = net(data, task_idx, mem_batch_size)
        else:
            spk_out, mem_out, conv_out = net(data, mem_batch_size)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out.detach().cpu())
        conv_rec.append(conv_out.reshape(1,conv_out.shape[0],conv_out.shape[1],conv_out.shape[2],conv_out.shape[3]))

    return torch.stack(spk_rec), torch.stack(mem_rec), torch.cat(conv_rec, 0)

def batch_accuracy(data_loader, net, num_steps, device, task_idx=None, task_labels=None, cam=False, cam_dir=None):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        data_loader = iter(data_loader)
        counter = 0
        for data, targets in data_loader:
            counter += 1
            img_data = data
            data = spikegen.rate_conv(data)
            data = data.to(device)
            
            if task_idx:
                targets = torch.where(targets == task_labels[task_idx][0], 0, 1)
            targets = targets.to(device)
            
            spk_rec, _, saved_forward = forward_pass(net, num_steps, data, task_idx)
            
            if cam:
                print('Computing CAM...')
                cams = compute_CAM(img_data[0], saved_forward, cam_dir)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc/total

def getForwardCAM(feature_conv):
    print('11111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
    print(feature_conv.shape)
    cam = feature_conv.sum(axis=0).sum(axis=0)
    cam = cam - np.min(cam)
    cam_img = cam / (np.max(cam) +1e-3)
    return cam_img


def compute_CAM(img, acts, cam_dir, show_flag=True):
    #print(f'1 {acts.shape}') #[30, 1, 12, 12, 12] = [n_steps,b,d,h,w]

    gamma = 0.5
    process = 0
    time = 0
    cam_save = 0
    cams = []
    previous_spike_time_list = []
    fig = plt.figure(figsize=(12,6))
    plt.imshow(img.permute(1, 2, 0), cmap='gray')
    plt.axis('off')
    plt.savefig(cam_dir+f'img.png', dpi=600)

    for l, activation in enumerate(acts):
        activation = activation # [1,12,12,12] = [b,d,h,w]
        previous_spike_time_list.append(activation)
        weight = 0

        for prev_t in range(len(previous_spike_time_list)):
            
            delta_t = time - previous_spike_time_list[prev_t] * prev_t
            weight +=  torch.exp(gamma * (-1) * delta_t)

        weighted_activation = weight.cuda() * activation
        weighted_activation = weighted_activation.data.cpu().numpy()
        overlay = getForwardCAM(weighted_activation)
        print('22222222222222222222222222222222222222222222222222222222222222222222222222222222222222')
        print(overlay)
        cam = skimage.transform.resize(overlay, (28, 28)) # 4x4 CAM
        cams.append(torch.from_numpy(cam).reshape(1,28,28))

        fig = plt.figure(figsize=(12,6))
        plt.imshow(img.permute(1, 2, 0), cmap='binary')
        plt.axis('off')
        plt.imshow(cam, alpha=0.5, cmap='jet')
        plt.savefig(cam_dir+f'actmap_{l}.png', dpi=600)
        
    return cams