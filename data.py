from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import copy


def load_data(dataset_name, img_size):
    
    if dataset_name == 'mnist':
        transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
        
        full_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, transform=transform)
    elif dataset_name == 'cifar10':
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
        
        full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    return full_dataset, test_dataset

def task_construction(task_labels, dataset_name, img_size, seed):
    full_dataset, test_dataset = load_data(dataset_name, img_size)
    
    full_dataset = split_dataset_by_labels(full_dataset, task_labels)
    
    train_dataset, val_dataset = [], []
    for dataset in full_dataset:
        length = len(dataset.targets)
        
        train_len = int(length * 0.9)
        val_len = int(length * 0.1)
        if length % 2 == 1:
            train_len += 1
            
        train_subset, val_subset = torch.utils.data.random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed))
        train_dataset.append(train_subset)
        val_dataset.append(val_subset)
    
    test_dataset = split_dataset_by_labels(test_dataset, task_labels)
    
    return train_dataset, val_dataset, test_dataset

def split_dataset_by_labels(dataset, task_labels):
    datasets = []
    
    for labels in task_labels:
        idx = np.in1d(dataset.targets, labels)
        splitted_dataset = copy.deepcopy(dataset)
        splitted_dataset.targets = splitted_dataset.targets[idx]
        splitted_dataset.data = splitted_dataset.data[idx]
        datasets.append(splitted_dataset)
        
    return datasets

def get_task_load_train(train_dataset,batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader

def get_task_load_test(test_dataset, batch_size):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return test_loader