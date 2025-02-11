# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    # Specify the config directory
    with initialize(config_path="."):
        cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_custom(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    image_size=1024,
    # num_maskmem=7,
):
    hydra_overrides = [
        # "++model._target_=rvt.mvt.sam2_train.sam2_custom.SAM2Custom",
        f"++model.image_size={image_size}",
        # f"++model.num_maskmem={num_maskmem}",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    # Specify the config directory
    with initialize(config_path="."):
        cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_custom_select(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    image_size=1024,
    num_maskmem=7,
    memory_in_dim=256,
    include_keys=None
):
    hydra_overrides = [
        "++model._target_=sam2act.mvt.sam2_train.sam2_custom.SAM2Custom",
        f"++model.image_size={image_size}",
        f"++model.num_maskmem={num_maskmem}",
        f"++model.memory_encoder.in_dim={memory_in_dim}",
        f"++model.memory_encoder.mask_downsampler.embed_dim={memory_in_dim}",
        f"++model.memory_encoder.fuser.layer.dim={memory_in_dim}",
        f"++model.memory_attention.d_model={memory_in_dim}",
        f"++model.memory_attention.layer.d_model={memory_in_dim}",
        f"++model.memory_attention.layer.self_attention.embedding_dim={memory_in_dim}",
        f"++model.memory_attention.layer.cross_attention.embedding_dim={memory_in_dim}",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    # Specify the config directory
    with initialize(config_path=".", version_base=None):
        cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint_select(model, ckpt_path, include_keys)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint_select(model, ckpt_path, include_keys=None):
    if ckpt_path is not None:
        # Load the checkpoint
        sd = torch.load(ckpt_path, map_location="cpu")["model"]

        # If specific keys are to be included, filter the state dictionary
        if include_keys is not None:
            sd = {k: v for k, v in sd.items() if any(k.startswith(key) for key in include_keys)}

        # Load the filtered state dictionary into the model
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)

        if missing_keys:
            logging.debug(f"Missing keys: {missing_keys}")  # Suppressed to debug level
        if unexpected_keys:
            logging.debug(f"Unexpected keys in checkpoint: {unexpected_keys}")  # Suppressed to debug level

        logging.info("Loaded checkpoint successfully")


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):
    hydra_overrides = [
        "++model._target_=sam2_train.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
