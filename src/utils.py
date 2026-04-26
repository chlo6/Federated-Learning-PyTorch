#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '/home/coder/Federated-Learning-PyTorch/data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                        transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                        transform=apply_transform)
        else:  # fmnist
            data_dir = '../data/fmnist/'
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))])  # FashionMNIST stats
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                transform=apply_transform)


        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights_fedbn(w):
    """
    FedBN-style aggregation: averages ONLY the trainable weights (conv, linear,
    BN gamma/beta), but EXCLUDES BN running statistics from aggregation.

    BN layers have 4 keys per layer:
      - weight (gamma)    ← trainable, DO average
      - bias (beta)       ← trainable, DO average
      - running_mean      ← running stat, DO NOT average
      - running_var       ← running stat, DO NOT average
      - num_batches_tracked ← counter, DO NOT average

    By keeping BN running stats LOCAL (not averaged), each client maintains
    its own distribution statistics. The global model only shares the
    learned feature transformation weights.

    This is what your professor meant by:
    "handle weight and normalization layer separately"
    "decouple into branch then avg branch by branch"
    """
    w_avg = copy.deepcopy(w[0])

    # Keys to skip during aggregation (BN running statistics)
    bn_stat_keys = ['running_mean', 'running_var', 'num_batches_tracked']

    for key in w_avg.keys():
        # Skip BN running statistics — keep them local (from client 0 / global model)
        if any(bn_key in key for bn_key in bn_stat_keys):
            continue

        # Average all other parameters (conv weights, linear weights, BN gamma/beta)
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))

    return w_avg


def average_weights_fedbn_weighted(w, data_sizes):
    """
    Weighted FedBN: weights each client's contribution by dataset size.
    More correct than simple averaging when clients have unequal data.
    Still skips BN running stats.

    Args:
        w: list of state_dicts from each client
        data_sizes: list of ints, number of training samples per client
    """
    total = sum(data_sizes)
    weights = [s / total for s in data_sizes]

    w_avg = copy.deepcopy(w[0])
    bn_stat_keys = ['running_mean', 'running_var', 'num_batches_tracked']

    # Initialize with first client weighted
    for key in w_avg.keys():
        if any(bn_key in key for bn_key in bn_stat_keys):
            continue
        w_avg[key] = w_avg[key] * weights[0]

    for key in w_avg.keys():
        if any(bn_key in key for bn_key in bn_stat_keys):
            continue
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * weights[i]

    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
