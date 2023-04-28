import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn

class Backbone(nn.Module):
    def __init__(self, beta, spike_grad, target_layer):
        super().__init__()
        
        self.target_layer = target_layer

        # Initialize layers
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)
        
        if self.target_layer == 1:
            sal_out = spk1

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)
        
        if self.target_layer == 2:
            sal_out = spk2

        return spk2, mem2, sal_out


class CilClassifier(nn.Module):
    def __init__(self, embed_dim, nb_classes, beta, spike_grad, batch_size, device):
        super().__init__()
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.beta = beta
        self.spike_grad = spike_grad
        self.device = device
        self.heads = nn.ModuleList([nn.Linear(embed_dim, nb_classes).to(device)])
        self.lifs = nn.ModuleList([snn.Leaky(beta=beta, spike_grad=spike_grad)])

    def __getitem__(self, index):
        return self.heads[index]

    def __len__(self):
        return len(self.heads)

    def forward(self, x, task_idx=None):
        x = x.view(self.batch_size, -1)

        mems = [lif.init_leaky() for lif in self.lifs]

        curs = [fc(x) for fc in self.heads]

        spks = []
        for i, lif in enumerate(self.lifs):
            spk, mem = lif(curs[i], mems[i]) # [B, 2]
            spks.append(spk)
            mems[i] = mem

        spk = torch.cat(spks, dim=1) # [B, (task_id+1)*2]
        mem = torch.cat(mems, dim=1)
        
        if task_idx:
            return spks[task_idx], mems[task_idx]
        
        return spk, mem

    def adaptation(self, nb_classes):
        self.heads.append(nn.Linear(self.embed_dim, nb_classes).to(self.device))
        self.lifs.append(snn.Leaky(beta=self.beta, spike_grad=self.spike_grad))


class CilModel(nn.Module):
    def __init__(self, beta, spike_grad, batch_size, device, target_layer=2):
        super(CilModel, self).__init__()
        
        self.beta = beta
        self.spike_grad = spike_grad
        self.batch_size = batch_size
        self.device = device
        
        self.backbone = Backbone(beta, spike_grad, target_layer=target_layer)
        self.feature_dim = 64*4*4
        self.fc = None

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, task_id=None):
        spk, mem, conv_out = self.backbone(x)
        spk, mem = self.fc(spk, task_id)
        return spk, mem, conv_out

    def copy(self):
        return copy.deepcopy(self)

    def prev_model_adaptation(self, nb_classes):
        if self.fc is None:
            self.fc = CilClassifier(self.feature_dim, nb_classes, self.beta, self.spike_grad, self.batch_size, self.device).to(self.device)
        else:
            self.fc.adaptation(nb_classes)

    def after_model_adaptation(self, nb_classes, task_idx):
        if task_idx > 0:
            self.weight_align(nb_classes)

    @torch.no_grad()
    def weight_align(self, nb_new_classes):
        w = torch.cat([head.weight.data for head in self.fc], dim=0)
        norms = torch.norm(w, dim=1)

        norm_old = norms[:-nb_new_classes]
        norm_new = norms[-nb_new_classes:]

        gamma = torch.mean(norm_old) / torch.mean(norm_new)
        self.fc[-1].weight.data = gamma * w[-nb_new_classes:]
        
class Net(nn.Module):
    def __init__(self, beta, spike_grad, batch_size, target_layer=2):
        super().__init__()
        
        self.target_layer = target_layer
        self.batch_size = batch_size

        # Initialize layers
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.head = nn.Linear(64*4*4, 10)
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, mem_batch_size=None):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem = self.lif.init_leaky()

        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)
        
        if self.target_layer == 1:
            sal_out = spk1

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)
        
        if self.target_layer == 2:
            sal_out = spk2
            
        if mem_batch_size:
            cur = self.head(spk2.view(mem_batch_size, -1))
        else:
            cur = self.head(spk2.view(self.batch_size, -1))
            
        spk, mem = self.lif(cur, mem)
        
        return spk, mem, sal_out
    
class ANN(nn.Module):
    def __init__(self, batch_size, target_layer=2):
        super().__init__()
        
        self.target_layer = target_layer
        self.batch_size = batch_size

        # Initialize layers
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.head = nn.Linear(64*4*4, 10)

    def forward(self, x, mem_batch_size=None):

        out = F.max_pool2d(self.conv1(x), 2)
        
        if self.target_layer == 1:
            sal_out = out

        out = F.max_pool2d(self.conv2(out), 2)
        
        if self.target_layer == 2:
            sal_out = spk2
            
        if mem_batch_size:
            out = self.head(out.view(mem_batch_size, -1))
        else:
            out = self.head(out.view(self.batch_size, -1))
                    
        return out, sal_out