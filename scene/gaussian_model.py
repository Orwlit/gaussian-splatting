#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


class GaussianModel:
    """_summary_
    - 点里存储的优化变量：
        变量名          | 意义          | 维度
        _xyz               三维坐标                             [num_pointcloud, 3]
        _features_dc       球谐系数: 0维的?                     [num_pointcloud, 1, 3]
        _features_rest     球谐系数: 1到3维的全部15个?           [num_pointcloud, 15, 3]
        _opacity            不透明度                            [num_pointcloud, 1]
        _scaling            xyz坐标的尺度?                      [num_pointcloud, 3]
        _rotation           各向异性协方差：四元数表示的旋转矩阵    [num_pointcloud, 4]
    """


    def setup_functions(self):
        """不同属性设置不同激活函数, 见论文 5.1 optimization
        """
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        """初始化属性, 从config读sh参数

        Args:
            sh_degree (int): _description_
        """
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        """捕获模型的当前状态, 包括参数和优化器状态。

        Returns:
            _type_: _description_
        """
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        """根据给定的模型参数和训练参数恢复模型状态。

        Args:
            model_args (_type_): _description_
            training_args (_type_): _description_
        """
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        """计算并返回协方差矩阵。

        Args:
            scaling_modifier (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        """增加球谐函数的度数, 用于更精细地表示颜色特征。增加当前活动的球谐函数度数, 直到达到最大度数。
        """
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        """从colmap/随即点云建立model

        Args:
            pcd (BasicPointCloud): 点云数据
            spatial_lr_scale (float): 空间学习率的缩放因子
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))


        """nn.Parameter相关知识:
        nn.Parameter 是 PyTorch 中的一个类, 用于定义可以被神经网络训练的参数。
        在 PyTorch 中, nn.Parameter 通常用于将张量 (tensor )标记为模型的可训练参数。
        """
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc =   nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        """根据给定的训练参数设置优化器。

        Args:
            training_args (_type_): 包含训练参数的对象
        """
        # ! 初始化
        # 设置场景中密集点的百分比。此值可能用于后续的密集化步骤。
        self.percent_dense = training_args.percent_dense
        # 初始化梯度累积器。这用于在反向传播过程中累积XYZ位置参数的梯度。
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # 初始化用于梯度归一化的分母。
        self.denom =              torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 创建参数列表, 其中包括模型的各个参数以及它们对应的学习率。
        l = [
            # 3D位置
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # 颜色？？
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # 球谐函数的颜色特征？学习率是_features_dc的1/20。
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            # 不透明度
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # xyz轴的尺度
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # 四元数旋转矩阵, 文中称为3D Gaussian的各向异性协方差 (anisotropic covariance )
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # 创建Adam优化器
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # 设置XYZ位置参数的学习率调度器, 该调度器会随着训练迭代的增加调整学习率。
        # 这个调度器的作用是在训练过程中动态调整位置参数的学习率, 以实现更有效的训练和收敛。
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        """根据迭代次数调整优化器的学习率, 实现学习率调度

        Args:
            iteration (_type_): 当前的迭代次数

        Returns:
            float: 学习率
        """
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        """保存模型的状态到PLY文件。

        Args:
            path (str): 要保存的ply文件路径
        """
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        """自适应高斯控制, 详情见论文 5.2 adaptive control of Gaussians
        """
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        """从PLY文件加载模型的状态

        Args:
            path (str): 要加载的ply文件路径
        """
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        """替换优化器中的参数张量
        @detail: 这通常在模型训练过程中的某些阶段需要改变已有参数的张量时使用。例如, 当需要调整某个参数的值, 但希望保留其在优化器中的历史状态 (比如动量项 )时, 可以使用这个函数。
                 这种方法通常用在复杂的训练场景中, 如模型微调、参数的动态调整等, 尤其在需要在保持优化器状态的前提下更新参数时非常有效。

        Args:
            tensor (torch.Tensor): 要在优化器中替换的新的 torch.Tensor 对象
            name (_type_): 要替换的参数的名称。这个名称用于在优化器的参数组中找到对应的参数

        Returns:
            _type_: _description_
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def _prune_optimizer(self, mask):
        """修剪（prune）优化器中的参数，即根据给定的掩码（mask）保留或删除参数的某些部分。

        Args:
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        # 初始化一个字典来存储修剪后的参数
        optimizable_tensors = {}

        # 遍历优化器中的每个参数组
        for group in self.optimizer.param_groups:
            # 获取当前参数的优化器状态（如果存在）
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                # 如果状态存在，根据掩码更新状态的动量项和平方梯度项
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                # 删除原有参数的状态
                del self.optimizer.state[group['params'][0]]

                # 应用掩码，创建新的 nn.Parameter，并更新优化器状态
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True))) # 过滤某些不需要的参数
                self.optimizer.state[group['params'][0]] = stored_state

                # 将修剪后的参数加入返回字典
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 如果状态不存在，直接应用掩码创建新的 nn.Parameter
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))  # 过滤某些不需要的参数
                optimizable_tensors[group["name"]] = group["params"][0]

        # 返回修剪后的参数字典
        return optimizable_tensors


    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """将新的张量（tensor）添加到优化器的参数组中。

        Args:
            tensors_dict (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        # 初始化一个字典来存储更新后的参数
        optimizable_tensors = {}
        
        # 遍历优化器中的每个参数组
        for group in self.optimizer.param_groups:
            # 确保每个参数组只包含一个参数
            assert len(group["params"]) == 1
            # 获取当前参数组名对应的扩展张量
            extension_tensor = tensors_dict[group["name"]]
            # 获取当前参数的优化器状态（如果存在）
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            # 如果状态存在，更新状态的动量项和平方梯度项
            if stored_state is not None:
            
                # 将扩展张量的零初始化状态与原状态拼接
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                # 删除原有参数的状态
                del self.optimizer.state[group['params'][0]]
                # 使用拼接后的新参数创建新的 nn.Parameter，并更新优化器状态
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)) # 拼接新参数
                self.optimizer.state[group['params'][0]] = stored_state

                # 将更新后的参数加入返回字典
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 如果状态不存在，直接使用拼接后的新参数创建新的 nn.Parameter
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)) # 拼接新参数
                optimizable_tensors[group["name"]] = group["params"][0]

        # 返回更新后的参数字典
        return optimizable_tensors



    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        """添加或更新高斯模型的点，并重置一些状态。

        Args:
            new_xyz (tensor): 新的xyz坐标。
            new_features_dc (tensor): 新的0阶球谐系数。
            new_features_rest (tensor): 新的1-3阶球谐系数。
            new_opacities (tensor): 新的不透明度。
            new_scaling (tensor): 新的缩放参数。
            new_rotation (tensor): 新的旋转参数。
        作用:
        - 将新的点集合与现有的模型参数结合。
        - 重置梯度累积和其他相关状态。
        """
        
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation
            }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        
        """针对over-reconstruction(grad>=threshold), 注意新的gaussian的计算

        Args:
            grads (tensor): 计算的梯度。
            grad_threshold (float): 用于分割的梯度阈值。
            scene_extent (float): 场景的大小或范围。
            N (int, optional): 每个原始点创建的新点数。Defaults to 2.

        作用:
        - 对于梯度大于阈值的点，按照N的数量创建新的高斯。
        - 保持其他特征（颜色、不透明度等）不变。
        - 更新高斯模型以包含这些新点。
        """        

        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        
        # ! 其他features,rotation,opacity不变
        new_features_dc =   self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """under-reconstruction (grad>=threshold) simple copy并沿梯度放置

        Args:
            grads (tensor): 计算的梯度。
            grad_threshold (float): 克隆的梯度阈值。
            scene_extent (float): 场景的大小或范围。

        作用:
        - 对于梯度大于阈值的点，创建它们的副本。
        - 更新高斯模型以包含这些克隆的点。
        """
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """clone + split(), 计算小于min_opacity的mask并删掉这些点, 详情参考5.2 Adaptive control of gaussians

        Args:
            max_grad (float): 用于密集化的最大梯度阈值。
            min_opacity (float): 用于修剪的最小不透明度。
            extent (float): 场景的大小或范围。
            max_screen_size (float): 最大屏幕尺寸限制。

        作用:
        - 克隆和分割满足条件的点。
        - 基于不透明度和屏幕尺寸修剪点。
        - 更新高斯模型以反映这些更改。
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        更新密集化统计数据。

        Args:
            viewspace_point_tensor (tensor): 视图空间点的张量。
            update_filter (tensor): 用于更新的过滤器。

        作用:
        - 累积用于密集化决策的梯度统计。
        - 更新指定点的梯度累积和计数器。
        """
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1