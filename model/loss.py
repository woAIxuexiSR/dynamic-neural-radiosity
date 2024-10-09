import torch
import torch.nn as nn
import numpy as np

EPS = 0.1

# lhs : [batch_size, 3], rhs : [batch_size, M, 3]
def l2_loss(lhs, rhs):
    rhs = rhs.mean(dim=1)
    return torch.mean((lhs - rhs) ** 2)

def normed_l2_loss(lhs, rhs):
    rhs = rhs.mean(dim=1)
    norm = (lhs + rhs).detach() / 2 + EPS
    return torch.mean(((lhs / norm) - (rhs / norm)) ** 2)

def normed_semi_l2_loss(lhs, rhs):
    rhs = rhs.mean(dim=1).detach()
    norm = (lhs + rhs).detach() / 2 + EPS
    return torch.mean(((lhs / norm) - (rhs / norm)) ** 2)

def l1_loss(lhs, rhs):
    rhs = rhs.mean(dim=1)
    return torch.mean(torch.abs(lhs - rhs))

def normed_l1_loss(lhs, rhs):
    rhs = rhs.mean(dim=1)
    norm = (lhs + rhs).detach() / 2 + EPS
    return torch.mean(torch.abs((lhs / norm) - (rhs / norm)))

def cross_l1_loss(lhs, rhs):
    M = rhs.shape[1]
    rhs1 = rhs[:, 0 : M // 2, :].mean(dim=1)
    rhs2 = rhs[:, M // 2 :, :].mean(dim=1)
    return torch.mean(torch.abs((lhs - rhs1) * (lhs - rhs2)))

def normed_cross_l1_loss(lhs, rhs):
    M = rhs.shape[1]
    rhs1 = rhs[:, 0 : M // 2, :].mean(dim=1)
    rhs2 = rhs[:, M // 2 :, :].mean(dim=1)
    norm1 = (lhs + rhs1).detach() / 2 + EPS
    norm2 = (lhs + rhs2).detach() / 2 + EPS
    return torch.mean(torch.abs((lhs / norm1 - rhs1 / norm1) * (lhs / norm2 - rhs2 / norm2)))

def cross_l2_loss(lhs, rhs):
    M = rhs.shape[1]
    rhs1 = rhs[:, 0 : M // 2, :].mean(dim=1)
    rhs2 = rhs[:, M // 2 :, :].mean(dim=1)
    return torch.mean(((lhs - rhs1) * (lhs - rhs2)) ** 2)

def normed_cross_l2_loss(lhs, rhs):
    M = rhs.shape[1]
    rhs1 = rhs[:, 0 : M // 2, :].mean(dim=1)
    rhs2 = rhs[:, M // 2 :, :].mean(dim=1)
    norm1 = (lhs + rhs1).detach() / 2 + EPS
    norm2 = (lhs + rhs2).detach() / 2 + EPS
    return torch.mean(((lhs / norm1 - rhs1 / norm1) * (lhs / norm2 - rhs2 / norm2)) ** 2)