import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import Tensor
# from torch.nn.parameter import Parameter
from sam2act.mvt.sam2_train.modeling.sam2_base import SAM2Base
# from safetensors import safe_open
# from safetensors.torch import save_file

# from icecream import ic


class _LoRA_qkv(nn.Module):
    """In SAM2 it is implemented as
    B, H, W, _ = x.shape
    # qkv with shape (B, H * W, 3, nHead, C)
    qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
    # q, k, v with shape (B, H * W, nheads, C)
    q, k, v = torch.unbind(qkv, 2)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q.cuda()
        self.linear_b_q = linear_b_q.cuda()
        self.linear_a_v = linear_a_v.cuda()
        self.linear_b_v = linear_b_v.cuda()
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features).cuda()

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv
    

class _LoRA_q_proj(nn.Module):
    """In SAM2 it is implemented as
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)

    # Separate into heads
    q = self._separate_heads(q, self.num_heads)
    k = self._separate_heads(k, self.num_heads)
    v = self._separate_heads(v, self.num_heads)
    """

    def __init__(
        self,
        q_proj: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
    ):
        super().__init__()
        self.q_proj = q_proj
        self.linear_a_q = linear_a_q.cuda()
        self.linear_b_q = linear_b_q.cuda()
        self.dim = q_proj.in_features
        self.w_identity = torch.eye(q_proj.in_features).cuda()

    def forward(self, x):
        q_proj = self.q_proj(x)  # B,C,H*W
        new_q = self.linear_b_q(self.linear_a_q(x))
        q_proj += new_q
        return q_proj

class _LoRA_k_proj(nn.Module):
    """In SAM2 it is implemented as
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)

    # Separate into heads
    q = self._separate_heads(q, self.num_heads)
    k = self._separate_heads(k, self.num_heads)
    v = self._separate_heads(v, self.num_heads)
    """

    def __init__(
        self,
        k_proj: nn.Module,
        linear_a_k: nn.Module,
        linear_b_k: nn.Module,
    ):
        super().__init__()
        self.k_proj = k_proj
        self.linear_a_k = linear_a_k.cuda()
        self.linear_b_k = linear_b_k.cuda()
        self.dim = k_proj.in_features
        self.w_identity = torch.eye(k_proj.in_features).cuda()

    def forward(self, x):
        k_proj = self.k_proj(x)  # B,C,H*W
        new_k = self.linear_b_k(self.linear_a_k(x))
        k_proj += new_k
        return k_proj

class _LoRA_v_proj(nn.Module):
    """In SAM2 it is implemented as
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)

    # Separate into heads
    q = self._separate_heads(q, self.num_heads)
    k = self._separate_heads(k, self.num_heads)
    v = self._separate_heads(v, self.num_heads)
    """

    def __init__(
        self,
        v_proj: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.v_proj = v_proj
        self.linear_a_v = linear_a_v.cuda()
        self.linear_b_v = linear_b_v.cuda()
        self.dim = v_proj.in_features
        self.w_identity = torch.eye(v_proj.in_features).cuda()

    def forward(self, x):
        v_proj = self.v_proj(x)  # B,C,H*W
        new_v = self.linear_b_v(self.linear_a_v(x))
        v_proj += new_v
        return v_proj
    

class LoRA_SAM2(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam2_model: SAM2Base, r: int, lora_layer=None):
        super(LoRA_SAM2, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            # self.lora_layer = list(range(len(sam2_model.image_encoder.trunk.blocks))) + list(range(len(sam2_model.memory_attention.layers)))  # Only apply lora to the image encoder and memory attention by default
            self.lora_layer = list(range(len(sam2_model.image_encoder.trunk.blocks)))
            # self.lora_layer = self.lora_layer[len(self.lora_layer) // 2:]

        # print(self.lora_layer)
            
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for name, param in sam2_model.named_parameters():
            # if "memory_attention" in name or "memory_encoder" in name or "mask_downsample" in name or "obj_ptr_proj" in name or "maskmem_tpos_enc_list" in name:
            # if "image_encoder" in name:
            # # if "image_encoder" in name or "memory_attention" in name:
            #     param.requires_grad = False
            # else:
            #     param.requires_grad = True
            param.requires_grad = False

        # Here, we do the surgery only on image encoder
        for t_layer_i, blk in enumerate(sam2_model.image_encoder.trunk.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        
        # for t_layer_i, layer in enumerate(sam2_model.memory_attention.layers):
        #     # If we only want few lora layer instead of all
        #     if t_layer_i not in self.lora_layer:
        #         continue
            
        #     # # add lora to self attention's q
        #     # w_self_attn_q_proj_linear = layer.self_attn.q_proj
        #     # self.dim = w_self_attn_q_proj_linear.in_features
        #     # w_self_attn_a_linear_q_proj = nn.Linear(self.dim, r, bias=False)
        #     # w_self_attn_b_linear_q_proj = nn.Linear(r, self.dim, bias=False)
        #     # self.w_As.append(w_self_attn_a_linear_q_proj)
        #     # self.w_Bs.append(w_self_attn_b_linear_q_proj)
        #     # layer.self_attn.q_proj = _LoRA_q_proj(
        #     #     w_self_attn_q_proj_linear,
        #     #     w_self_attn_a_linear_q_proj,
        #     #     w_self_attn_b_linear_q_proj,
        #     # )
            
        #     # # add lora to self attention's k
        #     # w_self_attn_k_proj_linear = layer.self_attn.k_proj
        #     # self.dim = w_self_attn_k_proj_linear.in_features
        #     # w_self_attn_a_linear_k_proj = nn.Linear(self.dim, r, bias=False)
        #     # w_self_attn_b_linear_k_proj = nn.Linear(r, self.dim, bias=False)
        #     # self.w_As.append(w_self_attn_a_linear_k_proj)
        #     # self.w_Bs.append(w_self_attn_b_linear_k_proj)
        #     # layer.self_attn.k_proj = _LoRA_k_proj(
        #     #     w_self_attn_k_proj_linear,
        #     #     w_self_attn_a_linear_k_proj,
        #     #     w_self_attn_b_linear_k_proj,
        #     # )

        #     # # add lora to self attention's v
        #     # w_self_attn_v_proj_linear = layer.self_attn.v_proj
        #     # self.dim = w_self_attn_v_proj_linear.in_features
        #     # w_self_attn_a_linear_v_proj = nn.Linear(self.dim, r, bias=False)
        #     # w_self_attn_b_linear_v_proj = nn.Linear(r, self.dim, bias=False)
        #     # self.w_As.append(w_self_attn_a_linear_v_proj)
        #     # self.w_Bs.append(w_self_attn_b_linear_v_proj)
        #     # layer.self_attn.v_proj = _LoRA_v_proj(
        #     #     w_self_attn_v_proj_linear,
        #     #     w_self_attn_a_linear_v_proj,
        #     #     w_self_attn_b_linear_v_proj,
        #     # )

        #     # add lora to cross attention's q
        #     w_cross_attn_q_proj_linear = layer.cross_attn_image.q_proj
        #     self.dim = w_cross_attn_q_proj_linear.in_features
        #     w_cross_attn_a_linear_q_proj = nn.Linear(self.dim, r, bias=False)
        #     w_cross_attn_b_linear_q_proj = nn.Linear(r, self.dim, bias=False)
        #     self.w_As.append(w_cross_attn_a_linear_q_proj)
        #     self.w_Bs.append(w_cross_attn_b_linear_q_proj)
        #     layer.cross_attn_image.q_proj = _LoRA_q_proj(
        #         w_cross_attn_q_proj_linear,
        #         w_cross_attn_a_linear_q_proj,
        #         w_cross_attn_b_linear_q_proj,
        #     )

        #     # # add lora to cross attention's k
        #     # w_cross_attn_k_proj_linear = layer.cross_attn_image.k_proj
        #     # self.dim_in = w_cross_attn_k_proj_linear.in_features
        #     # self.dim_out = w_cross_attn_k_proj_linear.out_features # note that in and out have different dim for k_proj in cross attn
        #     # w_cross_attn_a_linear_k_proj = nn.Linear(self.dim_in, r, bias=False)
        #     # w_cross_attn_b_linear_k_proj = nn.Linear(r, self.dim_out, bias=False)
        #     # self.w_As.append(w_cross_attn_a_linear_k_proj)
        #     # self.w_Bs.append(w_cross_attn_b_linear_k_proj)
        #     # layer.cross_attn_image.k_proj = _LoRA_k_proj(
        #     #     w_cross_attn_k_proj_linear,
        #     #     w_cross_attn_a_linear_k_proj,
        #     #     w_cross_attn_b_linear_k_proj,
        #     # )

        #     # add lora to cross attention's v
        #     w_cross_attn_v_proj_linear = layer.cross_attn_image.v_proj
        #     self.dim_in = w_cross_attn_v_proj_linear.in_features
        #     self.dim_out = w_cross_attn_v_proj_linear.out_features # note that in and out have different dim for v_proj in cross attn
        #     w_cross_attn_a_linear_v_proj = nn.Linear(self.dim_in, r, bias=False)
        #     w_cross_attn_b_linear_v_proj = nn.Linear(r, self.dim_out, bias=False)
        #     self.w_As.append(w_cross_attn_a_linear_v_proj)
        #     self.w_Bs.append(w_cross_attn_b_linear_v_proj)
        #     layer.cross_attn_image.v_proj = _LoRA_v_proj(
        #         w_cross_attn_v_proj_linear,
        #         w_cross_attn_a_linear_v_proj,
        #         w_cross_attn_b_linear_v_proj,
        #     )

        self.reset_parameters()
        self.sam2 = sam2_model


    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


    def get_sam2_model(self) -> SAM2Base:
        return self.sam2
    

    # def save_lora_parameters(self, filename: str) -> None:
    #     r"""Only safetensors is supported now.

    #     pip install safetensor if you do not have one installed yet.

    #     save both lora and fc parameters.
    #     """

    #     assert filename.endswith(".pt") or filename.endswith('.pth')

    #     num_layer = len(self.w_As)  # actually, it is half
    #     a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
    #     b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
    #     prompt_encoder_tensors = {}
    #     mask_decoder_tensors = {}

    #     # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
    #     if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
    #         state_dict = self.sam.module.state_dict()
    #     else:
    #         state_dict = self.sam.state_dict()
    #     for key, value in state_dict.items():
    #         if 'prompt_encoder' in key:
    #             prompt_encoder_tensors[key] = value
    #         if 'mask_decoder' in key:
    #             mask_decoder_tensors[key] = value

    #     merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
    #     torch.save(merged_dict, filename)

    # def load_lora_parameters(self, filename: str) -> None:
    #     r"""Only safetensors is supported now.

    #     pip install safetensor if you do not have one installed yet.\

    #     load both lora and fc parameters.
    #     """

    #     assert filename.endswith(".pt") or filename.endswith('.pth')

    #     state_dict = torch.load(filename)

    #     for i, w_A_linear in enumerate(self.w_As):
    #         saved_key = f"w_a_{i:03d}"
    #         saved_tensor = state_dict[saved_key]
    #         w_A_linear.weight = Parameter(saved_tensor)

    #     for i, w_B_linear in enumerate(self.w_Bs):
    #         saved_key = f"w_b_{i:03d}"
    #         saved_tensor = state_dict[saved_key]
    #         w_B_linear.weight = Parameter(saved_tensor)

    #     sam_dict = self.sam.state_dict()
    #     sam_keys = sam_dict.keys()

    #     # load prompt encoder
    #     prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
    #     prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
    #     prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
    #     sam_dict.update(prompt_encoder_new_state_dict)

    #     # load mask decoder
    #     mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
    #     mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
    #     mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
    #     sam_dict.update(mask_decoder_new_state_dict)
    #     self.sam.load_state_dict(sam_dict)


    # def forward(self, batched_input, multimask_output, image_size):
    #     return self.sam(batched_input, multimask_output, image_size)


    # def forward(self, x: Tensor) -> Tensor:
    #     return self.lora_vit(x)


class LoRA_SAM2_mem(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam2_model: SAM2Base, r: int, lora_layer=None):
        super(LoRA_SAM2_mem, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam2_model.image_encoder.trunk.blocks))) + list(range(len(sam2_model.memory_attention.layers)))  # Only apply lora to the image encoder and memory attention by default
            
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        
        # freeze
        # for name, param in sam2_model.named_parameters():
        #     # if "memory_attention" in name or "memory_encoder" in name or "mask_downsample" in name or "obj_ptr_proj" in name or "maskmem_tpos_enc_list" in name:
        #     # if "image_encoder" in name:
        #     # # if "image_encoder" in name or "memory_attention" in name:
        #     #     param.requires_grad = False
        #     # else:
        #     #     param.requires_grad = True
        #     param.requires_grad = False


        # Here, we do the surgery only on image encoder
        for t_layer_i, blk in enumerate(sam2_model.image_encoder.trunk.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        
        # for t_layer_i, layer in enumerate(sam2_model.memory_attention.layers):
        #     # If we only want few lora layer instead of all
        #     if t_layer_i not in self.lora_layer:
        #         continue
            
        #     # # add lora to self attention's q
        #     # w_self_attn_q_proj_linear = layer.self_attn.q_proj
        #     # self.dim = w_self_attn_q_proj_linear.in_features
        #     # w_self_attn_a_linear_q_proj = nn.Linear(self.dim, r, bias=False)
        #     # w_self_attn_b_linear_q_proj = nn.Linear(r, self.dim, bias=False)
        #     # self.w_As.append(w_self_attn_a_linear_q_proj)
        #     # self.w_Bs.append(w_self_attn_b_linear_q_proj)
        #     # layer.self_attn.q_proj = _LoRA_q_proj(
        #     #     w_self_attn_q_proj_linear,
        #     #     w_self_attn_a_linear_q_proj,
        #     #     w_self_attn_b_linear_q_proj,
        #     # )
            
        #     # # add lora to self attention's k
        #     # w_self_attn_k_proj_linear = layer.self_attn.k_proj
        #     # self.dim = w_self_attn_k_proj_linear.in_features
        #     # w_self_attn_a_linear_k_proj = nn.Linear(self.dim, r, bias=False)
        #     # w_self_attn_b_linear_k_proj = nn.Linear(r, self.dim, bias=False)
        #     # self.w_As.append(w_self_attn_a_linear_k_proj)
        #     # self.w_Bs.append(w_self_attn_b_linear_k_proj)
        #     # layer.self_attn.k_proj = _LoRA_k_proj(
        #     #     w_self_attn_k_proj_linear,
        #     #     w_self_attn_a_linear_k_proj,
        #     #     w_self_attn_b_linear_k_proj,
        #     # )

        #     # # add lora to self attention's v
        #     # w_self_attn_v_proj_linear = layer.self_attn.v_proj
        #     # self.dim = w_self_attn_v_proj_linear.in_features
        #     # w_self_attn_a_linear_v_proj = nn.Linear(self.dim, r, bias=False)
        #     # w_self_attn_b_linear_v_proj = nn.Linear(r, self.dim, bias=False)
        #     # self.w_As.append(w_self_attn_a_linear_v_proj)
        #     # self.w_Bs.append(w_self_attn_b_linear_v_proj)
        #     # layer.self_attn.v_proj = _LoRA_v_proj(
        #     #     w_self_attn_v_proj_linear,
        #     #     w_self_attn_a_linear_v_proj,
        #     #     w_self_attn_b_linear_v_proj,
        #     # )

        #     # add lora to cross attention's q
        #     w_cross_attn_q_proj_linear = layer.cross_attn_image.q_proj
        #     self.dim = w_cross_attn_q_proj_linear.in_features
        #     w_cross_attn_a_linear_q_proj = nn.Linear(self.dim, r, bias=False)
        #     w_cross_attn_b_linear_q_proj = nn.Linear(r, self.dim, bias=False)
        #     self.w_As.append(w_cross_attn_a_linear_q_proj)
        #     self.w_Bs.append(w_cross_attn_b_linear_q_proj)
        #     layer.cross_attn_image.q_proj = _LoRA_q_proj(
        #         w_cross_attn_q_proj_linear,
        #         w_cross_attn_a_linear_q_proj,
        #         w_cross_attn_b_linear_q_proj,
        #     )

        #     # # add lora to cross attention's k
        #     # w_cross_attn_k_proj_linear = layer.cross_attn_image.k_proj
        #     # self.dim_in = w_cross_attn_k_proj_linear.in_features
        #     # self.dim_out = w_cross_attn_k_proj_linear.out_features # note that in and out have different dim for k_proj in cross attn
        #     # w_cross_attn_a_linear_k_proj = nn.Linear(self.dim_in, r, bias=False)
        #     # w_cross_attn_b_linear_k_proj = nn.Linear(r, self.dim_out, bias=False)
        #     # self.w_As.append(w_cross_attn_a_linear_k_proj)
        #     # self.w_Bs.append(w_cross_attn_b_linear_k_proj)
        #     # layer.cross_attn_image.k_proj = _LoRA_k_proj(
        #     #     w_cross_attn_k_proj_linear,
        #     #     w_cross_attn_a_linear_k_proj,
        #     #     w_cross_attn_b_linear_k_proj,
        #     # )

        #     # add lora to cross attention's v
        #     w_cross_attn_v_proj_linear = layer.cross_attn_image.v_proj
        #     self.dim_in = w_cross_attn_v_proj_linear.in_features
        #     self.dim_out = w_cross_attn_v_proj_linear.out_features # note that in and out have different dim for v_proj in cross attn
        #     w_cross_attn_a_linear_v_proj = nn.Linear(self.dim_in, r, bias=False)
        #     w_cross_attn_b_linear_v_proj = nn.Linear(r, self.dim_out, bias=False)
        #     self.w_As.append(w_cross_attn_a_linear_v_proj)
        #     self.w_Bs.append(w_cross_attn_b_linear_v_proj)
        #     layer.cross_attn_image.v_proj = _LoRA_v_proj(
        #         w_cross_attn_v_proj_linear,
        #         w_cross_attn_a_linear_v_proj,
        #         w_cross_attn_b_linear_v_proj,
        #     )

        # freeze
        for name, param in sam2_model.named_parameters():
            # if "memory_attention" in name or "memory_encoder" in name or "mask_downsample" in name or "obj_ptr_proj" in name or "maskmem_tpos_enc_list" in name:
            if "image_encoder" in name:
            # if "image_encoder" in name or "memory_attention" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
            # param.requires_grad = False
            
        self.reset_parameters()
        self.sam2 = sam2_model


    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


    def get_sam2_model(self) -> SAM2Base:
        return self.sam2
    

if __name__ == "__main__":
    sam2 = build_sam2_multiview("sam2_hiera_b+", "/gscratch/krishna/hqfang/SAM-E/samE/mvt/sam2_train/checkpoints/sam2_hiera_base_plus.pt", device="cuda")

    output = sam2.image_encoder(torch.rand(size=(1, 3, 320, 320)).cuda())
    print(output.keys())
    print(output['vision_features'].shape)

    lora_sam2 = LoRA_SAM2(sam2, 4)
    output = lora_sam2.sam2.image_encoder(torch.rand(size=(1, 3, 320, 320)).cuda())
    print(output.keys())
    print(output['vision_features'].shape)

