
import torch
import numpy as np
from scipy.spatial import distance_matrix


def get_random_problems(batch_size, problem_size):
    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)
    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)
    if problem_size == 20:
        scaler = 3
    elif problem_size == 50:
        scaler = 4
    elif problem_size == 100:
        scaler = 5
    else:
        raise NotImplementedError
    total_n = 1+problem_size
    dm = np.zeros((batch_size, total_n, total_n),dtype=np.float32)
    for b in range(batch_size):
        depot = depot_xy[b].numpy()
        node = node_xy[b].numpy()
        nodes = np.concatenate([depot, node], axis=0)
        distance = distance_matrix(nodes,nodes) / scaler
        dm[b] = distance 
    # shape: (batch, Problem)
    # depot_xy, node_xy, distance_to_depot, distance_matrix
    return depot_xy, node_xy, torch.from_numpy(dm[:,0,1:]), torch.from_numpy(dm)


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data