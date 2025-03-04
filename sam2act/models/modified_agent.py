
from sam2act.models.sam2act_agent import SAM2Act_Agent,manage_loss_log,manage_eval_log
import sam2act.mvt.mvt_sam2 as mvt_sam2
import config as exp_cfg_mod
from sam2act.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
    DATA_FOLDER,
    DATA_FOLDER_MEM,
)
import torch,os
import sam2act.mvt.config as mvt_cfg_mod
from sam2act.mvt.augmentation import apply_se3_aug_con, apply_se3_aug_con_same, aug_utils
import sam2act.mvt.utils as mvt_utils
import sam2act.utils.rvt_utils as rvt_utils
import peract_colab.arm.utils as arm_utils
from torch.cuda.amp import autocast, GradScaler
import numpy as np

class SAM2Act_Agent2(SAM2Act_Agent):
    def __init__(self, *args, **kwargs):
        super(SAM2Act_Agent2, self).__init__(*args, **kwargs)

    def prepare_train_input(self,batch):
        assert batch['pc_fts'].shape[-1]==7
        assert batch['ee_poses'].shape[-1]==8
        assert batch['txt_embeds'].shape[0]==int(77*len(batch['txt_lens']))
        assert batch['txt_embeds'].shape[-1]==512
        bs = len(batch['txt_lens'])
        offsets=batch['offset'].cpu().numpy().tolist()
        offsets=[0]+offsets
        inp={
            'pc': None,'img_feat':None,'proprio':None,'lang_goal_embs':None,
            'wpt_local':None
        }
        inp['lang_goal_embs']=batch['txt_embeds'].reshape(bs,-1,512)
        inp['proprio']=batch['ee_poses']
        inp['pc']=[batch['pc_fts'][:, :3][offsets[i]:offsets[i+1]] for i in range(bs)]
        inp['img_feat']=[(batch['pc_fts'][:, -3:][offsets[i]:offsets[i+1]] + 1) / 2 for i in range(bs)]
        #reverse normalize rgb

        action_ignore_collisions = batch["gt_trajs_stop"].int()  # 转化成预测是否stop
        action_gripper_pose = batch['gt_trajs'][:,:-1].reshape(bs,7)  # (b, 7)
        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3)
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:]
        action_grip = batch['gt_trajs'][:,-1].reshape(bs).int()  # (b,)

        lang_goal_embs = inp['lang_goal_embs']
        tasks = None

        proprio = inp['proprio']  
        
        pc, img_feat = inp['pc'], inp['img_feat']
        out={
            "pc":pc,
            "img_feat":img_feat,
            "proprio":proprio,
            "lang_goal_embs":lang_goal_embs,
            "action_trans_con":action_trans_con,
            "action_rot":action_rot,
            "action_grip":action_grip,
            "action_ignore_collisions":action_ignore_collisions,
            "action_gripper_pose":action_gripper_pose
        }
        return out

    def prepare_act_input(self,batch):
        pass
    

    def __call__(self,batch, 
        backprop: bool = True,
        reset_log: bool = False,
    ) -> dict:
        # TO DO: 
        # from ['data_ids', 'pc_fts', 'pc_labels', 'pc_centroids', 'pc_radius', 'ee_poses', 'txt_embeds', 
        # 'gt_trajs', 'gt_trajs_stop', 'gt_trajs_disc_pos_probs', 'npoints_in_batch', 'offset', 
        # 'txt_lens', 'traj_lens', 'traj_masks']
        # get replay_sample: dict,[rot_grip_action_indicies,ignore_collisions,
        # gripper_pose,lang_goal_embs,low_dim_state]
        # need pc=pc,img_feat,proprio,lang_goal_embs,img_aug=0,wpt_local,rot_x_y
        # pc:list elements (n_pts,3),不等长
        # img_feat:同上，只包括颜色
        # proprio (bsz,4)
        # lang_goal_embs(bsz,77,embed_size)
        # wpt_local(bsz,3) float
        # rot_x_y(bsz,2) one-hot vector eg. [52, 36]

        # assert batch['pc_fts'].shape[-1]==7中间是高度
        # batch['pc_fts'][:, :3] -> pc in 621
        # batch['pc_fts'][:, -3:]-> rgb before 621
        # batch['gt_trajs'][:,0,:3] -> position
        # batch['gt_trajs'][:,0,3:-1] -> rot
        # batch['gt_trajs'][-1] -> open
        # txt_embeds(sum_txt_len,embed_size)
        # proprio用ee_poses替代，8维度，3维位置，4维旋转，1维开关按顺序来
        # batch['gt_trajs'][-1] -> obs.gripper_open
        # 时间信息使用batch['traj_lens']-stop后面有几个false
        bs = len(batch['txt_lens'])
        return_out = {}
        input_dict = self.prepare_train_input(batch)

        proprio = input_dict['proprio']
        action_gripper_pose = input_dict['action_gripper_pose']
        action_ignore_collisions = input_dict['action_ignore_collisions']
        pc = input_dict['pc']
        img_feat = input_dict['img_feat']
        action_grip = input_dict["action_grip"]
        lang_goal_embs = input_dict["lang_goal_embs"]
        action_trans_con = input_dict["action_trans_con"]
        action_rot = input_dict["action_rot"]

        assert action_grip.shape==(bs,)
        assert proprio.shape==(bs,8)
        assert action_gripper_pose.shape==(bs,7)
        assert action_ignore_collisions.shape==(bs,1)

        with torch.no_grad():
            
            if self._transform_augmentation and backprop:
                if not self._same_trans_aug_per_seq:
                    pc_lst=[]
                    action_trans_con_lst=[]
                    action_rot_lst=[]
                    for pc_i,action_gripper_pose_i in zip(pc,[x for x in action_gripper_pose]):
                        action_trans_con_i, action_rot_i, pc_i = apply_se3_aug_con(
                            pcd=pc_i.reshape(1,-1,3),
                            action_gripper_pose=action_gripper_pose_i.reshape(1,7),
                            bounds=torch.tensor(self.scene_bounds),
                            trans_aug_range=torch.tensor(self._transform_augmentation_xyz),
                            rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                        )
                        pc_lst.append(pc_i.reshape(-1,3))
                        action_trans_con_lst.append(torch.tensor(action_trans_con_i))
                        action_rot_lst.append(torch.tensor(action_rot_i))
                    pc = pc_lst
                    action_trans_con = torch.cat(action_trans_con_lst).to(pc[0].device)
                    action_rot = torch.cat(action_rot_lst).to(pc[0].device)
                else:
                    bs = pc.shape[0]
                    num_obs = self._num_maskmem + 1
                    num_seq = bs // num_obs

                    action_trans_con_after = []
                    action_rot_after = []
                    pc_after = []
                    for seq_idx in range(num_seq):
                        pc_i = pc[seq_idx*num_obs:seq_idx*num_obs+num_obs]
                        action_gripper_pose_i = action_gripper_pose[seq_idx*num_obs:seq_idx*num_obs+num_obs]
                        action_trans_con_i, action_rot_i, pc_i = apply_se3_aug_con_same(
                            pcd=pc_i,
                            action_gripper_pose=action_gripper_pose_i,
                            bounds=torch.tensor(self.scene_bounds),
                            trans_aug_range=torch.tensor(self._transform_augmentation_xyz),
                            rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                        )
                        action_trans_con_i = torch.tensor(action_trans_con_i).to(pc.device)
                        action_rot_i = torch.tensor(action_rot_i).to(pc.device)

                        action_trans_con_after.append(action_trans_con_i)
                        action_rot_after.append(action_rot_i)
                        pc_after.append(pc_i)
                    
                    action_trans_con = torch.cat(action_trans_con_after, dim=0)
                    action_rot = torch.cat(action_rot_after, dim=0)
                    pc = torch.cat(pc_after, dim=0)


            assert action_rot.shape==(bs,4)
            assert action_trans_con.shape==(bs,3)
            # TODO: vectorize
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot
            #剔除工作范围外的点,pc是列表每个元素(num_pt,3),各个元素不等长,并且只在这里存在剔除点，前面都是，(bsz,n_pts,3)的tensor
            # pc, img_feat = rvt_utils.move_pc_in_bound(
            #     pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            # )
            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)
            
            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None
        with autocast(enabled=self.amp):
            (
                action_rot_x_one_hot,
                action_rot_y_one_hot,
                action_rot_z_one_hot,
                action_grip_one_hot,  # (bs, 2)
                action_collision_one_hot,  # (bs, 2)
            ) = self._get_one_hot_expert_actions(
                bs, action_rot, action_grip, action_ignore_collisions, device=self._device
            )

            if self.rot_ver == 1:
                rot_x_y = torch.cat(
                    [
                        action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                        action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                    ],
                    dim=-1,
                )
                if self.rot_x_y_aug != 0:
                    # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
                    rot_x_y += torch.randint(
                        -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
                    ).to(rot_x_y.device)
                    rot_x_y %= self._num_rotation_classes

            # hm_gt = self.get_gt_hm(
            #     wpt_local, dyn_cam_info, dims=(bs, nc, h, w)
            # )

            out = self._network(
                pc=pc,
                img_feat=img_feat,
                proprio=proprio,
                lang_emb=lang_goal_embs,
                img_aug=img_aug,
                wpt_local=wpt_local if self._network.training else None,
                rot_x_y=rot_x_y if self.rot_ver == 1 else None,
                # hm_gt=hm_gt,
            )

            q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
                out, dims=(bs, nc, h, w)
            )
            pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
                out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info
            )
            continuous_action = np.concatenate(
                (
                    pred_wpt[0].cpu().numpy(),
                    pred_rot_quat[0],
                    pred_grip[0].cpu().numpy(),
                    pred_coll[0].cpu().numpy(),
                )
            )
            continuous_action = np.expand_dims(continuous_action, axis=1)
            action_trans = self.get_action_trans(
                wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
            )

        loss_log = {}
        if backprop:
            with autocast(enabled=self.amp):
                # cross-entropy loss
                trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()
                rot_loss_x = rot_loss_y = rot_loss_z = 0.0
                grip_loss = 0.0
                collision_loss = 0.0
                if not self.use_memory:
                    if self.add_rgc_loss:
                        rot_loss_x = self._cross_entropy_loss(
                            rot_q[
                                :,
                                0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                            ],
                            action_rot_x_one_hot.argmax(-1),
                        ).mean()

                        rot_loss_y = self._cross_entropy_loss(
                            rot_q[
                                :,
                                1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                            ],
                            action_rot_y_one_hot.argmax(-1),
                        ).mean()

                        rot_loss_z = self._cross_entropy_loss(
                            rot_q[
                                :,
                                2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                            ],
                            action_rot_z_one_hot.argmax(-1),
                        ).mean()

                        grip_loss = self._cross_entropy_loss(
                            grip_q,
                            action_grip_one_hot.argmax(-1),
                        ).mean()

                        collision_loss = self._cross_entropy_loss(
                            collision_q, action_collision_one_hot.argmax(-1)
                        ).mean()

                    total_loss = (
                        trans_loss
                        + rot_loss_x
                        + rot_loss_y
                        + rot_loss_z
                        + grip_loss
                        + collision_loss
                    )

                else:

                    total_loss = trans_loss


            self._optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()

            # self.scaler.unscale_(self._optimizer)
            # torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=0.1)

            self.scaler.step(self._optimizer)
            self.scaler.update()
            self._lr_sched.step()

            loss_log = {
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": rot_loss_x.item() if not self.use_memory else None,
                "rot_loss_y": rot_loss_y.item() if not self.use_memory else None,
                "rot_loss_z": rot_loss_z.item() if not self.use_memory else None,
                "grip_loss": grip_loss.item() if not self.use_memory else None,
                "collision_loss": collision_loss.item() if not self.use_memory else None,
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)
        
        return continuous_action,return_out
    @torch.no_grad()
    def _eval(self,batch):
        bs = len(batch['txt_lens'])
        return_out = {}
        input_dict = self.prepare_train_input(batch)

        proprio = input_dict['proprio']
        action_gripper_pose = input_dict['action_gripper_pose']
        action_ignore_collisions = input_dict['action_ignore_collisions']
        pc = input_dict['pc']
        img_feat = input_dict['img_feat']
        action_grip = input_dict["action_grip"]
        lang_goal_embs = input_dict["lang_goal_embs"]
        action_trans_con = input_dict["action_trans_con"]
        action_rot = input_dict["action_rot"]
        # TODO: Vectorize
        pc_new = []
        rev_trans = []
        for _pc in pc:
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size
        dyn_cam_info = None

        out = self._network(
            pc=pc,
            img_feat=img_feat,
            proprio=proprio,
            lang_emb=lang_goal_embs,
            img_aug=0,  # no img augmentation while acting
        )
        wpt = [x[:3] for x in action_trans_con]

        wpt_local = []
        rev_trans = []
        for _pc, _wpt in zip(pc, wpt):
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                _wpt,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            wpt_local.append(a.unsqueeze(0))
            rev_trans.append(b)

        wpt_local = torch.cat(wpt_local, axis=0)
        q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
            out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=True
        )
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info
        )
        action_trans = self.get_action_trans(
            wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
        )
        (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot, 
            action_collision_one_hot, 
        ) = self._get_one_hot_expert_actions(
            bs, action_rot.cpu(), action_grip.cpu(), action_ignore_collisions.cpu(), device="cpu"
        )
        action_rot_x_one_hot = action_rot_x_one_hot.to(self._device)
        action_rot_y_one_hot = action_rot_y_one_hot.to(self._device)
        action_rot_z_one_hot = action_rot_z_one_hot.to(self._device)
        action_grip_one_hot = action_grip_one_hot.to(self._device)
        action_collision_one_hot = action_collision_one_hot.to(self._device)
        # for batched eval
        continuous_action = torch.cat(
            [
                pred_wpt.cpu(),
                torch.tensor(pred_rot_quat),
                pred_grip.cpu(),
                pred_coll.cpu(),
            ],1
        ).numpy()

        trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()
        rot_loss_x = rot_loss_y = rot_loss_z = 0.0
        grip_loss = 0.0
        collision_loss = 0.0
        rot_loss_x = self._cross_entropy_loss(
            rot_q[
                :,
                0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
            ],
            action_rot_x_one_hot.argmax(-1),
        ).mean()

        rot_loss_y = self._cross_entropy_loss(
            rot_q[
                :,
                1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
            ],
            action_rot_y_one_hot.argmax(-1),
        ).mean()

        rot_loss_z = self._cross_entropy_loss(
            rot_q[
                :,
                2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
            ],
            action_rot_z_one_hot.argmax(-1),
        ).mean()

        grip_loss = self._cross_entropy_loss(
            grip_q,
            action_grip_one_hot.argmax(-1),
        ).mean()

        collision_loss = self._cross_entropy_loss(
            collision_q, action_collision_one_hot.argmax(-1)
        ).mean()

        total_loss = (
            trans_loss
            + rot_loss_x
            + rot_loss_y
            + rot_loss_z
            + grip_loss
            + collision_loss
        )


        loss_log = {
            "total_loss": total_loss.item(),
            "trans_loss": trans_loss.item(),
            "rot_loss_x": rot_loss_x.item() if not self.use_memory else None,
            "rot_loss_y": rot_loss_y.item() if not self.use_memory else None,
            "rot_loss_z": rot_loss_z.item() if not self.use_memory else None,
            "grip_loss": grip_loss.item() if not self.use_memory else None,
            "collision_loss": collision_loss.item() if not self.use_memory else None,
            "lr": self._optimizer.param_groups[0]["lr"],
        }
        return_out.update(loss_log)
        return continuous_action,return_out

def get_logdir(cmd_args, exp_cfg):
    exp = exp_cfg.exp_id + '_' + exp_cfg.exp_name
    log_dir = os.path.join(cmd_args.log_dir, exp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def load_agent(cmd_args):
    #TODO: distributed
    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    if cmd_args.exp_cfg_path != "":
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))
    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    if cmd_args.mvt_cfg_path != "":
        mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
    if cmd_args.mvt_cfg_opts != "":
        mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))
    device="cuda:0"
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    sam2act = mvt_sam2.MVT_SAM2(
        renderer_device=device,
        rank=0,
        **mvt_cfg,
    ).to(device)
    log_dir = get_logdir(cmd_args, exp_cfg)
    TRAINING_ITERATIONS = 100
    EPOCHS = exp_cfg.epochs
    agent = SAM2Act_Agent2(
        network=sam2act,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        add_lang=mvt_cfg.add_lang,
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{log_dir}/test_run/",
        cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS,
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )
    agent.build(training=True, device=device)
    return agent
    
