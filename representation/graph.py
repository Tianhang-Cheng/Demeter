
import torch
import torch.nn as nn
import open3d as o3d
import tqdm
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from utils.pca import NodePCA
from utils.constant import *
from utils.rotation_pytorch3d import quaternion_to_matrix, matrix_to_quaternion, standardize_quaternion
from utils.pcd import grid_to_mesh, get_min_dist, compute_frenet_serret_frame
from utils.plot import generate_cylinder_along_curve_batch, draw_tree_with_colors
from utils.graph import find_layers_and_paths
from representation.primitive import CatmullRomSurface, CatmullRomCurve

np.random.seed(42)
# generaet [0, 255] 3 channels colors
color_map = np.random.randint(0, 255, (256, 3)) / 255

def _nn_Parameter(tensor, requires_grad=True):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    return nn.Parameter(tensor.clone().float().detach().requires_grad_(requires_grad))

def _detach(x):
    return x.detach().cpu().numpy()

def interpolate_polyline(L, control_points):
    """
    Differentiable interpolation along a polyline with uniform segment lengths.
    
    Args:
        L: torch.Tensor of shape [...] with values in [0, 1], normalized length along polyline
        control_points: torch.Tensor of shape [n, 3], control points of the polyline
    
    Returns:
        torch.Tensor of shape [..., 3], interpolated positions on the polyline
    """
    n = control_points.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 control points")
    
    # Number of segments
    num_segments = n - 1
    
    # Scale L to segment space [0, num_segments]
    segment_pos = L * num_segments
    
    # Clamp to valid range to handle edge cases
    segment_pos = torch.clamp(segment_pos, 0, num_segments - 1e-6)
    
    # Get the segment index (which segment we're in)
    segment_idx = torch.floor(segment_pos).long()
    
    # Get the local parameter within the segment [0, 1]
    local_t = segment_pos - segment_idx.float()
    
    # Handle the batch dimensions properly
    segment_idx = segment_idx.item()
    p0 = control_points[segment_idx]
    p1 = control_points[segment_idx + 1]
    result = p0 + local_t * (p1 - p0)
    
    return result, segment_idx, local_t

def slerp_quaternion(q1, q2, t):
    """
    Spherical linear interpolation between two quaternions.
    
    Args:
        q1, q2: quaternions as tensors of shape (..., 4) [w, x, y, z]
        t: interpolation parameter in [0, 1]
    
    Returns:
        Interpolated quaternion
    """
    # Ensure quaternions are normalized
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)
    
    # Compute dot product
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)
    
    # If dot product is negative, negate one quaternion to take shorter path
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)
    
    # If quaternions are very close, use linear interpolation
    DOT_THRESHOLD = 0.9995
    if torch.any(dot > DOT_THRESHOLD):
        result = q1 + t * (q2 - q1)
        return result / torch.norm(result, dim=-1, keepdim=True)
    
    # Calculate angle between quaternions
    theta_0 = torch.acos(torch.clamp(dot, -1, 1))
    sin_theta_0 = torch.sin(theta_0)
    
    theta = theta_0 * t
    sin_theta = torch.sin(theta)
    
    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q1 + s1 * q2


class PlantGraphFixedTopology(nn.Module):
    def __init__(self, 
                 layers: edict=None,
                 classes: edict=None,
                 parents: edict=None,

                 pca_stem_3d: NodePCA=None,
                 pca_leaf_3d: NodePCA=None,
                 pca_leaf_2d: NodePCA=None,
                 
                 stem_3d_info_cp_local: edict=None, # viz
                 stem_3d_info_thickness: edict=None, # shape
                 stem_3d_info_deform_coeff: edict=None, # deformation
                 stem_3d_info_s: edict=None, # articulation (scale)
                 stem_3d_info_M_quat: edict=None, # articulation (rotation)
                 
                 leaf_3d_info_cp_local: edict=None, # viz
                 leaf_3d_info_shape_coeff: edict=None, # shape
                 leaf_3d_info_deform_coeff: edict=None, # deformation
                 leaf_3d_info_s: edict=None, # articulation (scale)
                 leaf_3d_info_M_quat: edict=None, # articulation (rotation)

                 node_length_along_parent_stem: edict=None, # articulation (translation)

                 species='soybean'):
        super().__init__()

        self.classes = classes
        self.parents = parents
        self.species = species

        assert pca_stem_3d is not None, 'we need pca_stem to decode stem shape'
        assert pca_leaf_3d is not None, 'we need pca_leaf to decode leaf shape'
        self.pca_stem_3d = pca_stem_3d
        self.pca_leaf_3d = pca_leaf_3d

        # Some initialization
        self.layers = find_layers_and_paths(self.parents)[0]
        self.main_stem = str(self.layers[0][0])
        self.stem_key = []
        self.leaf_key = []

        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]
            for child_k in layer:
                child_k = str(child_k)
                if self.classes[child_k] == FLOWER_CLASS or self.classes[child_k] == FRUIT_CLASS:
                    continue
                if self.classes[child_k] == STEM_CLASS:
                    self.stem_key.append(child_k)
                elif self.classes[child_k] == LEAF_CLASS:
                    self.leaf_key.append(child_k)

        # hardcoded for now
        self.leaf_w = 43
        self.leaf_h = 45

        # empty parameters, just need some function from it
        self._surface = CatmullRomSurface(species=species, shape_pca=pca_leaf_2d)
        self._curve = CatmullRomCurve()

        # load parameters from instance fitting
        if stem_3d_info_cp_local is None:
            return
        
        for layer_idx in range(len(layers)):
            layer = layers[layer_idx]
            for child_k in layer:
                child_k = str(child_k)

                if classes[child_k] == FLOWER_CLASS or classes[child_k] == FRUIT_CLASS:
                    continue

                if classes[child_k] == STEM_CLASS:
                    self.stem_key.append(child_k)
                elif classes[child_k] == LEAF_CLASS:
                    self.leaf_key.append(child_k)

                if classes[child_k] == STEM_CLASS:
                    self.register_parameter(name=f'cp_w_local_{child_k}', param=_nn_Parameter(stem_3d_info_cp_local[child_k]))
                    self.register_parameter(name=f'thickness_{child_k}', param=_nn_Parameter(stem_3d_info_thickness[child_k]))
                    self.register_parameter(name=f'deform_{child_k}', param=_nn_Parameter(stem_3d_info_deform_coeff[child_k]))
                    self.register_parameter(name=f'scale_{child_k}', param=_nn_Parameter(1/stem_3d_info_s[child_k]))
                    self.register_parameter(name=f'M_quat_{child_k}', param=_nn_Parameter(stem_3d_info_M_quat[child_k]))
                elif classes[child_k] == LEAF_CLASS:
                    self.register_parameter(name=f'cp_w_local_{child_k}', param=_nn_Parameter(leaf_3d_info_cp_local[child_k]))
                    self.register_parameter(name=f'shape_{child_k}', param=_nn_Parameter(leaf_3d_info_shape_coeff[child_k]))
                    self.register_parameter(name=f'deform_{child_k}', param=_nn_Parameter(leaf_3d_info_deform_coeff[child_k]))
                    self.register_parameter(name=f'scale_{child_k}', param=_nn_Parameter(1/leaf_3d_info_s[child_k]))
                    self.register_parameter(name=f'M_quat_{child_k}', param=_nn_Parameter(leaf_3d_info_M_quat[child_k]))
                # common parameters
                self.register_parameter(name=f'length_{child_k}', param=_nn_Parameter(node_length_along_parent_stem[child_k]))

    def draw_topology(self):
        print('leaf node is green, stem node is brown, flower is pink, fruit is darkgreen')
        draw_tree_with_colors(self.parents, c=self.classes)

    def coeff_to_leaf(self, shape_coeff, deform_coeff):
        a, b, c = self._surface.invert(self.pca_leaf_3d.decode(deform_coeff).reshape(43, 45, 3))
        return self._surface.evaluate(main_rotation=a, sub_rotation_l=b, sub_rotation_r=c, shape_coeff=shape_coeff)[0]
    
    def coeff_to_stem(self, deform_coeff):
        cp = self.pca_stem_3d.decode(deform_coeff).reshape(-1, 3)
        return cp

    def __len__(self):
        return len(self.stem_key) + len(self.leaf_key)
    
    def generate(self, output_format='point', color='gray', **kwargs):
        """
        output_format: 
            'point': return a tensor of 3d points
            'seg_point': return 2 tensors of 3d points, one for stem and one for leaf
            'instance': return a dict of individual point clouds
            'mesh': return a o3d.geometry.TriangleMesh
            'instance_mesh': return a dict of individual meshes
            'seg_mesh': return a tuple of two o3d.geometry.TriangleMesh
        kwargs:
            align_global: bool, align the main stem to global axis
        """

        # for editing
        # current_node_full_articulation = kwargs.get('current_node_full_articulation', True)
        # current_node_full_shape = kwargs.get('current_node_full_shape', True)
        # current_node_full_deform = kwargs.get('current_node_full_deform', True)
        stem_blend_weight = kwargs.get('stem_blend_weight', 1.0)
        leaf_shape_blend_weight = kwargs.get('leaf_shape_blend_weight', 1.0)
        leaf_deform_blend_weight = kwargs.get('leaf_deform_blend_weight', 1.0)
        max_processed = kwargs.get('max_processed', 1000)
        edit_id = kwargs.get('edit_id', -1)


        layers = self.layers
        classes = self.classes
        main_stem = self.main_stem
        parents = self.parents

        assert layers is not None and classes is not None and parents is not None, 'Please provide layers, classes and parents for inference'

        axis_on_parent = edict()
        pcd_stem = edict()
        pcd_leaf = edict()

        geometries = []
        geometries_dict = {}
        geometries_M_p_dict = {}
        geometries_M_dict = {}
        geometries_offset_dict = {}

        geometries_dict_stem = {}
        offset_losses = torch.tensor(0.0).float().cuda()

        # sort layers by layer index
        layers = {k: layers[k] for k in sorted(layers.keys())}

        # get mean scale
        stem_mean_scale = 0
        leaf_mean_scale = 0
        for k in self.stem_key:
            stem_mean_scale += self.__getattr__(f'scale_{k}')
        for k in self.leaf_key:
            leaf_mean_scale += self.__getattr__(f'scale_{k}')
        stem_mean_scale /= len(self.stem_key)
        leaf_mean_scale /= len(self.leaf_key)

        count = -1 # count number of nodes processed
        for layer_idx in layers:
            layer = layers[layer_idx]
            for child_k in layer:
                # filter out flower and fruit
                child_k = str(child_k)
                xp = str(parents[child_k])
                if classes[child_k] == FLOWER_CLASS or classes[child_k] == FRUIT_CLASS:
                    continue
                
                # limit the number of nodes processed for partial visualization
                count += 1
                if count >= max_processed:
                    break

                is_current_processed = (count == edit_id)
                
                # get relative information
                s = self.__getattr__(f'scale_{child_k}')
                M = quaternion_to_matrix(self.__getattr__(f'M_quat_{child_k}'))
                
                if is_current_processed:
                    if classes[child_k] == STEM_CLASS:
                        s = stem_mean_scale
                    elif classes[child_k] == LEAF_CLASS:
                        s = leaf_mean_scale
                
                if is_current_processed:
                    M = torch.eye(3).float().cuda()

                if child_k != main_stem:
                    l = self.__getattr__(f'length_{child_k}')
                    l = torch.clip(l, 0.0, 1.0)
                    if is_current_processed:
                        l = torch.tensor(0.0).float().cuda()

                    _pp = pcd_stem[xp]
                    _la = axis_on_parent[xp]

                    offset, _idx_lower, local_t  = interpolate_polyline(l, _pp)
                    _idx_upper = _idx_lower + 1

                    _la_lower = standardize_quaternion(matrix_to_quaternion(_la[_idx_lower]))
                    _la_upper = standardize_quaternion(matrix_to_quaternion(_la[_idx_upper]))
                    M_p = quaternion_to_matrix(slerp_quaternion(_la_lower, _la_upper, local_t))
                
                else:
                    M_p = torch.eye(3).float().cuda()
                    offset = torch.tensor([0, 0, 0]).float().cuda()
                
                geometries_offset_dict[child_k] = offset

                # calculate stem
                if classes[child_k] == STEM_CLASS:
                    if child_k == main_stem:
                        self.global_M = _detach(M).copy()
                    if kwargs.get('align_global', False) and child_k == main_stem:
                        M = torch.eye(3).float().cuda() # override M, aligned in world space

                    # if not self.skip_stem_pca_encoding:
                    #     cp_w_local = self.pca_stem.decode(cp_w_local).reshape(-1, 3)
                    #     cp_w_local = cp_w_local - cp_w_local[0]
                    # else:
                    #     cp_w_local = _curve.evaluate(rotation=cp_w_local, num_points=2, w=deform_coeff_stem)[0]
                    #     cp_w_local = cp_w_local.reshape(-1, 3)
                        
                    if 'uniform_thickness' in kwargs:
                        thickness = torch.tensor(kwargs['uniform_thickness']).float().cuda()
                    else:
                        thickness = self.__getattr__(f'thickness_{child_k}')
                    
                    thickness = torch.clip(thickness, min=0.0004)

                    # cp = self.__getattr__(f'cp_w_local_{child_k}') @ M @ M_p
                    # if not current_node_full_deform and is_current_processed:
                    #     cp = self.coeff_to_stem(stem_blend_weight * self.__getattr__(f'deform_{child_k}')) @ M @ M_p
                    # else:
                    # cp = self.coeff_to_stem(stem_blend_weight * self.__getattr__(f'deform_{child_k}')) @ M @ M_p
                    cp = self.coeff_to_stem((stem_blend_weight if is_current_processed else 1.0) * self.__getattr__(f'deform_{child_k}')) @ M @ M_p

                    cp = cp * s
                    p = cp # no need to interpolate
                    p = p + offset

                    axis_on_parent[child_k] = compute_frenet_serret_frame(cp, repeats=1)[0:p.shape[0]]
                    pcd_stem[child_k] = p.reshape(-1, 3)

                    if 'instance' in output_format or 'mesh' in output_format:
                        if is_current_processed:
                            p_viz = generate_cylinder_along_curve_batch(p, 0.0002, k=32) # FIXME: hardcoded thickness
                        else:
                            p_viz = generate_cylinder_along_curve_batch(p, thickness, k=32)
                        geometries_dict_stem[child_k] = p_viz
                    if 'mesh' in output_format:
                        if color == 'instance':
                            p_mesh = grid_to_mesh(p_viz, color=color_map[int(child_k) % 256])
                        else:
                            if is_current_processed:
                                p_mesh = grid_to_mesh(p_viz, color='orange')
                            else:
                                p_mesh = grid_to_mesh(p_viz, color=color)
                        geometries.append(p_mesh)
                        geometries_dict[child_k] = p_mesh
                        geometries_M_p_dict[child_k] = M_p
                        geometries_M_dict[child_k] = M

                # calculate leaf
                if classes[child_k] == LEAF_CLASS:

                    # cp_w_local = cp_w_local.reshape(self.leaf_w, self.leaf_h, 3)
                    # leaf_recon = self.coeff_to_leaf((leaf_shape_blend_weight if (not current_node_full_shape and is_current_processed) else 1) * self.__getattr__(f'shape_{child_k}'), 
                    #                                 (leaf_deform_blend_weight if (not current_node_full_deform and is_current_processed) else 1) * self.__getattr__(f'deform_{child_k}'))
                    # leaf_recon = self.coeff_to_leaf(leaf_shape_blend_weight * self.__getattr__(f'shape_{child_k}'), 
                    #                                 (leaf_deform_blend_weight * self.__getattr__(f'deform_{child_k}')))
                    leaf_recon = self.coeff_to_leaf((leaf_shape_blend_weight if (is_current_processed) else 1) * self.__getattr__(f'shape_{child_k}'), 
                                                    (leaf_deform_blend_weight if (is_current_processed) else 1) * self.__getattr__(f'deform_{child_k}'))                           
                    
                    # fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(cp_w_local[:,:,0].detach().cpu().numpy(), cp_w_local[:,:,1].detach().cpu().numpy(), cp_w_local[:,:,2].detach().cpu().numpy(), s=0.5)
                    # ax.scatter(leaf_recon[...,0].detach().cpu().numpy(), leaf_recon[...,1].detach().cpu().numpy(), leaf_recon[...,2].detach().cpu().numpy(), s=0.5, c='red')
                    # plt.show()

                    cp = leaf_recon @ M @ M_p
                    p = cp * s + offset
                    pcd_leaf[child_k] = p.reshape(-1, 3)

                    if 'mesh' in output_format:
                        if color == 'instance':
                            p_mesh = grid_to_mesh(p, color=color_map[int(child_k) % 256])
                        else:
                            # if is_current_processed and not all_the_same:
                            #     color = 'orange' # highlight the last processed node
                            if is_current_processed:
                                p_mesh = grid_to_mesh(p, color='orange')
                            else:
                                p_mesh = grid_to_mesh(p, color=color)
                        # p_mesh = p_mesh.subdivide_loop(number_of_iterations=2)
                        geometries.append(p_mesh)
                        geometries_dict[child_k] = p_mesh
                        geometries_M_p_dict[child_k] = M_p
                        geometries_M_dict[child_k] = M

        if output_format == 'instance_mesh':
            return geometries_dict

        if output_format == 'instance_mesh_full':
            return geometries_dict, geometries_M_p_dict, geometries_M_dict, geometries_offset_dict

        if output_format == 'seg_mesh' or output_format == 'seg_instance_mesh':
            merge_leaf_geometries = None
            merge_stem_geometries = None
            stem_geometries = []
            leaf_geometries = []
            for k, v in geometries_dict.items():
                if classes[k] == STEM_CLASS:
                    stem_geometries.append(copy.deepcopy(v))
                    if merge_stem_geometries is None:
                        merge_stem_geometries = v
                    else:
                        merge_stem_geometries += v
                elif classes[k] == LEAF_CLASS:
                    leaf_geometries.append(copy.deepcopy(v))
                    if merge_leaf_geometries is None:
                        merge_leaf_geometries = v
                    else:
                        merge_leaf_geometries += v
            if output_format == 'seg_mesh':
                return merge_stem_geometries, merge_leaf_geometries
            if output_format == 'seg_instance_mesh':
                return stem_geometries, leaf_geometries

        if output_format == 'mesh':
            merge_geometries = None
            for g in geometries:
                if merge_geometries is None:
                    merge_geometries = g
                else:
                    merge_geometries += g
            if kwargs.get('ref_pcd', None) is not None:
                ref_pcd = kwargs['ref_pcd']
                # assign the color of the ref_pcd to the mesh by the nearest point
                vertices = np.asarray(merge_geometries.vertices) 
                idx = get_min_dist(vertices, np.array(ref_pcd.points))[1]
                vertex_colors = np.array(ref_pcd.colors)[idx]
                merge_geometries.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            # if kwargs.get('align_global', False):
            #     merge_geometries.rotate(self.global_M_p.T, center=(0,0,0))
            return merge_geometries

        if output_format == 'instance':
            return pcd_stem, pcd_leaf, offset_losses
        
        if output_format == 'instance2':
            return pcd_stem, pcd_leaf, geometries_dict_stem

        p_list1 = [pcd_stem[k] for k in pcd_stem]
        p_list2 = [pcd_leaf[k] for k in pcd_leaf]
        p_list1 = torch.cat(p_list1, dim=0)
        p_list2 = torch.cat(p_list2, dim=0)
        
        if output_format == 'seg_point':
            return p_list1, p_list2

        p_list = torch.cat([p_list1, p_list2], dim=0)
        
        if output_format == 'point':
            return p_list

        return p_list
    
    def fit(self, p_ref, n_iter=1000, lr=1e-4, mode='cdist', **kwargs):
        """
        mode: 
            'cdist'
            'seg_cdist'
            'p2p'
            'finetune'
        """
        assert mode in ['cdist', 'seg_cdist', 'p2p', 'finetune']

        def chamfer_loss_a2b(dist):
            return torch.min(dist, dim=1).values 
        def chamfer_loss_b2a(dist):
            return torch.min(dist, dim=0).values

        # chamfer_loss = lambda dist: chamfer_loss_a2b(dist)[0] + chamfer_loss_b2a(dist)[0]

        params = []
        for layer_id, layer in self.layers.items():
            # layer_lr = lr * (len(layer)-layer_id) / len(layer) * 3
            layer_lr = lr

            for child_k in layer:
                child_k = str(child_k)

                if self.classes[child_k] == STEM_CLASS:
                    # params.append({'params': [self.__getattr__(f'deform_{child_k}')], 'lr': layer_lr})
                    params.append({'params': [self.__getattr__(f'scale_{child_k}')], 'lr': layer_lr})
                    if child_k != self.main_stem:
                        params.append({'params': [self.__getattr__(f'length_{child_k}')], 'lr': layer_lr })
                        params.append({'params': [self.__getattr__(f'M_quat_{child_k}')], 'lr': layer_lr})
                elif self.classes[child_k] == LEAF_CLASS:
                    params.append({'params': [self.__getattr__(f'length_{child_k}')], 'lr': layer_lr})
                    params.append({'params': [self.__getattr__(f'M_quat_{child_k}')], 'lr': layer_lr})
        
        optimizer = torch.optim.AdamW(params)

        # lr_end_scale = 1
        # gamma = lr_end_scale ** (1 / n_iter)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        scheduler = None

        for i in tqdm.tqdm(range(n_iter)):

            if mode == 'seg_cdist':
                p = self.generate(output_format='seg_point')
            elif mode == 'finetune':
                p = self.generate(output_format='instance', first_iter=(i==0))
                # p = self.generate(output_format='instance')
            else:
                p = self.generate(output_format='point')

            if mode == 'p2p':
                # p and p_ref are the same size
                loss = torch.square(p-p_ref).mean()
            elif mode == 'finetune':
                pcd_stem, pcd_leaf, offset_losses = p
                
                stem_fit_loss = torch.tensor(0.0).float().cuda()
                leaf_fit_loss = torch.tensor(0.0).float().cuda()

                n_leaf = len(pcd_leaf)
                for k in pcd_leaf:
                    p = pcd_leaf[k]
                    p_gt = p_ref[k]
                    
                    cur_loss = torch.square(p - p_gt).mean() # normalize by number of points 

                    leaf_fit_loss = leaf_fit_loss + cur_loss
                
                ignore_stem_loss = False
                if not ignore_stem_loss:
                    n_stem = len(pcd_stem)
                    for k in pcd_stem:
                        p = pcd_stem[k]
                        p_gt = p_ref[k]

                        cur_loss = torch.square(p - p_gt).mean() 

                        stem_fit_loss = stem_fit_loss + cur_loss

                loss = stem_fit_loss + leaf_fit_loss + offset_losses

            else:
                raise ValueError('Invalid mode')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            

            if i % 2 == 0:
                if mode != 'finetune':
                    print('Iter {}, loss {:.4e}'.format(i, loss.item()))
                else:
                    print('Iter {}, stem loss {:.4e}, leaf loss {:.4e}'.format(i, stem_fit_loss.item(), leaf_fit_loss.item()))

            if i == n_iter - 1:
                for k in self.stem_key + self.leaf_key:
                    self.__getattr__(f'M_quat_{k}').data = standardize_quaternion(self.__getattr__(f'M_quat_{k}').data)
                if mode == 'finetune':
                    for k in self.stem_key:
                        if self.parents[k] != -1:
                            # make sure the stem thickness is not larger than its parent
                            parent_thickness = self.__getattr__(f'thickness_{str(self.parents[k])}')
                            cur_thickness = self.__getattr__(f'thickness_{k}')
                            cur_thickness = min(cur_thickness, parent_thickness)
                            self.__getattr__(f'thickness_{k}').data = cur_thickness
        return p
    
    def dump_node_sequence(self):
        raise NotImplementedError

    def sequence_to_topology(self, node_sequence_type):
        raise NotImplementedError

    def save(self, path):
        torch.save(self.state_dict(), path)
        print('Model saved at', path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model_dict = self.state_dict()

        for k, v in checkpoint.items():
            if k not in model_dict:
                self.register_parameter(k, torch.nn.Parameter(v))

        # Now reload all keys
        self.load_state_dict(checkpoint, strict=False)