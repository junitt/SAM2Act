# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch

from torch.nn.init import trunc_normal_

from .modeling.sam2_base import NO_OBJ_SCORE, SAM2Base

from sam2act.mvt.sam2_train.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames


class SAM2Custom(SAM2Base):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(
        self,
        num_maskmem: int = 7,
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # whether to also clear non-conditioning memory of the surrounding frames (only effective when `clear_non_cond_mem_around_input` is True).
        clear_non_cond_mem_for_multi_obj=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_maskmem = num_maskmem
        
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        
        self.use_obj_ptrs_in_encoder = False
        self.use_mask_input_as_output_without_sam = True

        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)

    # def _encode_new_memory(
    #     self,
    #     current_vision_feats,
    #     feat_sizes,
    #     pred_masks_high_res,
    #     object_score_logits,
    #     is_mask_from_pts,
    #     if_sigmoid,
    # ):
    #     """Encode the current image and its prediction into a memory feature."""
    #     B = current_vision_feats[-1].size(1)  # batch size on this frame
    #     C = self.hidden_dim
    #     H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
    #     # top-level feature, (HW)BC => BCHW
    #     pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
    #     if self.non_overlap_masks_for_mem_enc and not self.training:
    #         # optionally, apply non-overlapping constraints to the masks (it's applied
    #         # in the batch dimension and should only be used during eval, where all
    #         # the objects come from the same video under batch size 1).
    #         pred_masks_high_res = self._apply_non_overlapping_constraints(
    #             pred_masks_high_res
    #         )
    #     # scale the raw mask logits with a temperature before applying sigmoid
    #     # binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
    #     if not if_sigmoid:
    #         mask_for_mem = (pred_masks_high_res > 0).float()
    #     else:
    #         # apply sigmoid on the raw mask logits to turn them into range (0, 1)
    #         mask_for_mem = torch.sigmoid(pred_masks_high_res)

    #     # apply scale and bias terms to the sigmoid probabilities
    #     if self.sigmoid_scale_for_mem_enc != 1.0:
    #         mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
    #     if self.sigmoid_bias_for_mem_enc != 0.0:
    #         mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
    #     maskmem_out = self.memory_encoder(
    #         pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
    #     )
    #     maskmem_features = maskmem_out["vision_features"]
    #     maskmem_pos_enc = maskmem_out["vision_pos_enc"]
    #     # add a no-object embedding to the spatial memory to indicate that the frame
    #     # is predicted to be occluded (i.e. no object is appearing in the frame)
    #     if self.no_obj_embed_spatial is not None:
    #         is_obj_appearing = (object_score_logits > 0).float()
    #         maskmem_features += (
    #             1 - is_obj_appearing[..., None, None]
    #         ) * self.no_obj_embed_spatial[..., None, None].expand(
    #             *maskmem_features.shape
    #         )

    #     return maskmem_features, maskmem_pos_enc