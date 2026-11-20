"""
General Utils for Models

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
from itertools import chain


@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.no_grad()
def bincount2offset(bincount):
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.no_grad()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def my_segment_coo(input, batch_after_match, reduce="mean"):

    unique_indexes = batch_after_match.unique()
    input_agg = []
    if reduce=="mean":
        for index in unique_indexes:
            input_agg.append(input[batch_after_match==int(index)].mean())

    elif reduce=="min":
        for index in unique_indexes:
            input_agg.append(input[batch_after_match==int(index)].min(dim=0)[0])

    output = torch.stack(input_agg)

    return output