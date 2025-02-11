# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import torch
import torch.distributed as dist
import random


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def setup_multinode():
    # Set the URL for communication
    dist_url = "env://" # default

    # Retrieve world_size, rank and local_rank
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ['LOCAL_RANK'])

    # Initialize the process group
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)
    
    torch.cuda.set_device(local_rank)


def cleanup():
    dist.destroy_process_group()
