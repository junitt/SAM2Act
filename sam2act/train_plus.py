# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import time
import tqdm
import random
import yaml
import argparse

from collections import defaultdict
from contextlib import redirect_stdout

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import config as exp_cfg_mod
import sam2act.models.sam2act_agent as sam2act_agent
import sam2act.utils.ddp_utils as ddp_utils
import sam2act.mvt.config as mvt_cfg_mod

import sam2act.mvt.mvt_sam2 as mvt_sam2
from sam2act.models.sam2act_agent import print_eval_log, print_loss_log
from sam2act.utils.get_dataset import get_dataset, get_dataset_temporal
from sam2act.utils.rvt_utils import (
    TensorboardManager,
    short_name,
    get_num_feat,
    load_agent,
    load_agent_only_model,
    load_agent_only_model_exclude,
    RLBENCH_TASKS,
)
from sam2act.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
    DATA_FOLDER,
    DATA_FOLDER_MEM,
)

import wandb
from sam2act.utils.wandb_utils import wandb_init


def get_model_size(model):
    """
    Calculate the size of a PyTorch model in bytes.
    """
    param_size = 0
    trainable_param_size = 0
    param_num = 0
    trainable_para_num = 0
    for param in model.parameters():
        param_num += param.nelement() 
        param_size += param.nelement() * param.element_size()
        trainable_para_num += param.nelement() if param.requires_grad else 0
        trainable_param_size += param.nelement() * param.element_size() if param.requires_grad else 0
        
    
    print(f'{model.__class__.__name__}\'s parameter size: {param_size/1024/1024}MB')
    print(f'{model.__class__.__name__}\'s trainable parameter size: {trainable_param_size/1024/1024}MB')
    
    print(f'{model.__class__.__name__}\'s parameter num: {param_num/1000/1000}M')
    print(f'{model.__class__.__name__}\'s trainable parameter num: {trainable_para_num/1000/1000}M')


# new train takes the dataset as input
def train(agent, dataset, training_iterations, log_iter, rank=0, node_rank=0, ifwandb=False):
    agent.train()
    log = defaultdict(list)

    data_iter = iter(dataset)
    iter_command = range(training_iterations)

    for iteration in tqdm.tqdm(
        iter_command, disable=(rank != 0), position=0, leave=True
    ):

        raw_batch = next(data_iter)
        batch = {
            k: v.to(agent._device)
            for k, v in raw_batch.items()
            if type(v) == torch.Tensor
        }
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"] = raw_batch["lang_goal"]
        update_args = {
            "step": iteration,
        }
        update_args.update(
            {
                "replay_sample": batch,
                "backprop": True,
                "reset_log": (iteration == 0),
                "eval_log": False,
            }
        )
        return_out = agent.update(**update_args)

        if (iteration + 1) % 100 == 0 and rank == 0:
            loss_log = agent.loss_log

            total_loss_avg = sum(loss_log['total_loss'][-100:]) / len(loss_log['total_loss'][-100:])
            trans_loss_avg = sum(loss_log['trans_loss'][-100:]) / len(loss_log['trans_loss'][-100:])

            print(f"total loss: {total_loss_avg} | trans loss: {trans_loss_avg}")

            if ifwandb and node_rank == 0:
                wandb.log(data = {
                                    'total_loss': loss_log['total_loss'][iteration],
                                    'trans_loss': loss_log['trans_loss'][iteration],
                                    # 'rot_loss_x': loss_log['rot_loss_x'][iteration],
                                    # 'rot_loss_y': loss_log['rot_loss_y'][iteration],
                                    # 'rot_loss_z': loss_log['rot_loss_z'][iteration],
                                    # 'grip_loss': loss_log['grip_loss'][iteration],
                                    # 'collision_loss': loss_log['collision_loss'][iteration],
                                    'lr': loss_log['lr'][iteration],
                                    }, 
                            step = log_iter)

        log_iter += 1

    if rank == 0:
        log = print_loss_log(agent)

    return log


def save_agent(agent, path, epoch):
    model = agent._network
    optimizer = agent._optimizer
    lr_sched = agent._lr_sched

    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "lr_sched_state": lr_sched.state_dict(),
        },
        path,
    )


def get_tasks(exp_cfg):
    parsed_tasks = exp_cfg.tasks.split(",")
    if parsed_tasks[0] == "all":
        tasks = RLBENCH_TASKS
    else:
        tasks = parsed_tasks
    return tasks


def get_logdir(cmd_args, exp_cfg):
    exp = exp_cfg.exp_id + '_' + exp_cfg.exp_name
    log_dir = os.path.join(cmd_args.log_dir, exp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir):
    with open(f"{log_dir}/exp_cfg_plus.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(exp_cfg.dump())

    with open(f"{log_dir}/mvt_cfg_plus.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(mvt_cfg.dump())

    args = cmd_args.__dict__
    with open(f"{log_dir}/args_plus.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file)


# def experiment(rank, cmd_args, devices, port):
def experiment(cmd_args, devices, rank, node_rank, world_size):
    """experiment.

    :param rank:
    :param cmd_args:
    :param devices: list or int. if list, we use ddp else not
    """
    # device = devices[rank]
    # device = f"cuda:{device}"
    device = f"cuda:{rank % world_size}"

    ddp = world_size > 1
    # ddp_utils.setup(rank, world_size=len(devices), port=port)

    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    if cmd_args.exp_cfg_path != "":
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))

    if ddp:
        print(f"Running DDP on rank {rank}.")

    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id

    exp_cfg.peract.lr *= world_size * exp_cfg.bs
    # if cmd_args.exp_cfg_opts != "":
    #     exp_cfg.exp_id += f"_{short_name(cmd_args.exp_cfg_opts)}"
    # if cmd_args.mvt_cfg_opts != "":
    #     exp_cfg.exp_id += f"_{short_name(cmd_args.mvt_cfg_opts)}"

    if rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
    exp_cfg.freeze()

    # Things to change
    BATCH_SIZE_TRAIN = exp_cfg.bs
    NUM_TRAIN = exp_cfg.demo
    # to match peract, iterations per epoch
    TRAINING_ITERATIONS = int(exp_cfg.train_iter // (exp_cfg.bs * world_size))
    EPOCHS = exp_cfg.epochs
    TRAIN_REPLAY_STORAGE_DIR = "replay_temporal/replay_train"
    TRAIN_REPLAY_STORAGE_DIR_MEM = "replay_temporal_memory/replay_train"
    # TEST_REPLAY_STORAGE_DIR = "replay/replay_val"
    log_dir = get_logdir(cmd_args, exp_cfg)
    tasks = get_tasks(exp_cfg)
    if rank == 0:
        print("Training on {} tasks: {}".format(len(tasks), tasks))

    # if exp_cfg.agent == "our":
    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    if cmd_args.mvt_cfg_path != "":
        mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
    if cmd_args.mvt_cfg_opts != "":
        mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

    mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
    mvt_cfg.freeze()

    # for maintaining backward compatibility
    assert mvt_cfg.num_rot == exp_cfg.peract.num_rotation_classes, print(
        mvt_cfg.num_rot, exp_cfg.peract.num_rotation_classes
    )

    t_start = time.time()
    get_dataset_func = lambda: get_dataset_temporal(
        tasks,
        BATCH_SIZE_TRAIN,
        None,
        # TRAIN_REPLAY_STORAGE_DIR,
        TRAIN_REPLAY_STORAGE_DIR_MEM,
        None,
        # DATA_FOLDER,
        DATA_FOLDER_MEM,
        NUM_TRAIN,
        None,
        cmd_args.refresh_replay,
        device,
        num_workers=exp_cfg.num_workers,
        only_train=True,
        sample_distribution_mode=exp_cfg.sample_distribution_mode,
        num_maskmem=mvt_cfg.num_maskmem,
        rank=rank,
    )
    train_dataset, _ = get_dataset_func()
    t_end = time.time()

    if rank == 0:
        print("Created Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

    if exp_cfg.agent == "our":

        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        sam2act = mvt_sam2.MVT_SAM2(
            renderer_device=device,
            rank=rank,
            **mvt_cfg,
        ).to(device)
        if rank == 0:
            get_model_size(sam2act)
        if ddp:
            sam2act = DDP(sam2act, device_ids=[device], find_unused_parameters=True)

        agent = sam2act_agent.SAM2Act_Agent(
            network=sam2act,
            image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
            add_lang=mvt_cfg.add_lang,
            stage_two=mvt_cfg.stage_two,
            rot_ver=mvt_cfg.rot_ver,
            scene_bounds=SCENE_BOUNDS,
            cameras=CAMERAS,
            log_dir=f"{log_dir}/test_run/",
            cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS,
            use_memory=mvt_cfg.use_memory,
            num_maskmem=mvt_cfg.num_maskmem,
            **exp_cfg.peract,
            **exp_cfg.rvt,
        )
        agent.build(training=True, device=device)
    else:
        assert False, "Incorrect agent"

    start_epoch = 0
    end_epoch = EPOCHS
    if exp_cfg.resume != "":
        agent_path = exp_cfg.resume

        if rank == 0:
            print(f"Recovering model and checkpoint from {exp_cfg.resume}")

        epoch = load_agent(agent_path, agent, only_epoch=False)
        start_epoch = epoch + 1

    elif os.path.exists(f'{log_dir}/model_plus_last.pth'):
        
        agent_path = f'{log_dir}/model_plus_last.pth'
        if rank == 0:
            print(f"resume from checkpoint")
        
        epoch = load_agent(agent_path, agent, only_epoch=False)
        if rank == 0:
            print(f"Recovering model and checkpoint from {agent_path}, model epoch: {epoch}")
        start_epoch = epoch + 1

    
    elif os.path.exists(f'{log_dir}/model_last.pth'):
        
        agent_path = f'{log_dir}/model_last.pth'
        if rank == 0:
            print(f"resume from checkpoint")
        
        # epoch = load_agent_only_model(agent_path, agent, only_epoch=False)
        epoch = load_agent_only_model_exclude(agent_path, agent, only_epoch=False, exclude_keys=['memory_attention', 'memory_encoder', 'maskmem_tpos_enc'])
        if rank == 0:
            print(f"Recovering model and checkpoint from {agent_path}")
        
    dist.barrier()

    if exp_cfg.wandb and rank == 0 and node_rank == 0:
        mode = os.getenv("WANDB_MODE", "online")
        key = os.getenv("WANDB_API_KEY")
        wandb_init(exp_cfg, mode, key)

    if rank == 0:
        ## logging unchanged values to reproduce the same setting
        temp1 = exp_cfg.peract.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost()
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
        exp_cfg.peract.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze()
        # tb = TensorboardManager(log_dir)


    if rank == 0:
        print("Start training ...", flush=True)

    i = start_epoch
    log_iter = 0
    while True:
        if i == end_epoch:
            break

        if rank == 0:
            print(f"Rank [{rank}], Epoch [{i}]: Training on train dataset")

        out = train(agent, train_dataset, TRAINING_ITERATIONS, log_iter, rank, node_rank, ifwandb=exp_cfg.wandb)

        # if rank == 0:
        #     tb.update("train", i, out)

        if rank == 0 and node_rank == 0:
            # TODO: add logic to only save some models
            save_agent(agent, f"{log_dir}/model_plus_{i}.pth", i)
            save_agent(agent, f"{log_dir}/model_plus_last.pth", i)
        i += 1
        log_iter += TRAINING_ITERATIONS

    if rank == 0:
        # tb.close()
        print("[Finish]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())

    parser.add_argument("--refresh_replay", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--mvt_cfg_path", type=str, default="")
    parser.add_argument("--exp_cfg_path", type=str, default="")

    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")

    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--with-eval", action="store_true", default=False)

    cmd_args = parser.parse_args()
    del (
        cmd_args.entry
    )  # hack for multi processing -- removes an argument called entry which is not picklable

    devices = cmd_args.device.split(",")
    devices = [int(x) for x in devices]

    # port = (random.randint(0, 3000) % 3000) + 27000
    # mp.spawn(experiment, args=(cmd_args, devices, port), nprocs=len(devices), join=True)


    # ddp_utils.setup_multinode()

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

    try:
        # Pass the LOCAL_RANK to the experiment function
        local_rank = int(os.environ["LOCAL_RANK"])
        experiment(cmd_args, devices, local_rank, rank, world_size)  # Adjust your experiment function to accept the local rank
    finally:
        ddp_utils.cleanup()

    
