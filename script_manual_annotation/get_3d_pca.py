import sys
sys.path.append('../Plant')

from utils.pca import NodePCA
import numpy as np
import glob
import torch
import matplotlib.pyplot as plt
from script_fitting.fitted import fitted
from utils.pcd import grid_to_mesh, grid_to_normals
import tqdm
import os
import open3d as o3d
from utils.plot import load_view_point

red = np.array([255, 51, 153]) / 255 
green = np.array([130, 253, 121]) / 255
blue = np.array([51, 153, 255]) / 255
brown = np.array([165, 42, 42]) / 255


def train_3d_pca(species: str):

    folder = f'/home/tianhang/data/{species}/pcd_processed'

    instance_list = fitted[species]

    # load leaf data
    leaf_3d = []
    for instance in instance_list:
        leaf_folder = f'{folder}/{instance}/leaf_template_deform'
        leaf_paths = glob.glob(f'{leaf_folder}/*.pth')
        print(f'Loading {len(leaf_paths)} leaf paths from {leaf_folder}')
        for leaf_path in leaf_paths:
            leaf_3d.append(torch.load(leaf_path, weights_only=True))
    
    leaf_3d = torch.stack(leaf_3d, dim=0) # (n, 43, 45, 3)
    b = leaf_3d.shape[0]
    print(f'Loaded {b} leaves.')


    if species == 'rose':
        n_components = 10
    else:
        n_components = 0.995
    leaf_pca = NodePCA(n_components=n_components) # 0.995
    scores = leaf_pca.train(leaf_3d.reshape(b, -1)) # (n, 43*45*3) -> (n, n_components)
    leaf_pca.save(f'/home/tianhang/data/{species}/3d_leaf_pca.pth')

    # mean_shape = leaf_pca.data_mean.reshape(43, 45, 3).cpu().numpy()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # viz = mean_shape
    # # viz = leaf_3d.mean(0).detach().cpu().numpy()
    # leaf_3d_viz = leaf_3d.detach().cpu().numpy()
    # for i in range(leaf_3d_viz.shape[0]):
    #     viz = leaf_3d_viz[i]
    #     ax.scatter(viz[..., 0], viz[..., 1], viz[..., 2], s=1)
    #     # for j in range(viz.shape[0]):
    #     #     ax.plot(viz[j, :, 0], viz[j, :, 1], viz[j, :, 2], linewidth=1.5, color='green', alpha=0.3)
    # # ax.scatter(viz[..., 0], viz[..., 1], viz[..., 2], s=1)
    # plt.show()

    """
    visualize PCA components as 3D mesh using plt
    """
    # output_3d_mesh = False
    # n_frames = 50
    # if output_3d_mesh:
    #     for i in tqdm.tqdm(range(len(leaf_pca.components))):

    #         components_viz_dir = f'/home/tianhang/data/folio/Folio Leaf Dataset/Folio/{species}_rotated_mask/leaf_deformation_pca_viz/{i}'
    #         os.makedirs(components_viz_dir, exist_ok=True)

    #         component_scores = scores[:, i]
    #         sigma = np.std(component_scores)

    #         for index, t in enumerate(np.linspace(-2, 2, 50)):
    #             data_modified = np.zeros_like(scores[0].copy()).astype(np.float32)
    #             data_modified[i] += t * sigma
    #             shape_info_recover = leaf_pca.decode(data_modified)

    #             shape_info_recover = shape_info_recover.reshape(43, 45, 3)

    #             # fig = plt.figure()
    #             # ax = fig.add_subplot(111, projection='3d')
    #             # ax.plot_surface(shape_info_recover[..., 0], shape_info_recover[..., 1], shape_info_recover[..., 2], cmap='viridis')
    #             # ax.set_title(f'PCA Component {i}, t={t:.2f}')
    #             # plt.axis('off')
    #             # plt.show()

    #             shape_mesh = grid_to_mesh(shape_info_recover) # open3d mesh
    #             save_path = os.path.join(components_viz_dir, f'{str(index).zfill(6)}.ply')
    #             o3d.io.write_triangle_mesh(save_path, shape_mesh)
    #             print(f'Saved {save_path}')
    #             if i > 3:
    #                 break
    #             save_path2 = os.path.join(f'/home/tianhang/code/Demeter-project.github.io/static/interpolation_3d_leaf/{species}/pca{i+1}', f'{str(index).zfill(6)}.ply')
    #             o3d.io.write_triangle_mesh(save_path2, shape_mesh)
                # _=1
    
    """
    visualize Leaf PCA components using open3d
    """
    output_3d_plot = False
    n_frames = 5
    max_range = 3.0
    h_leaf, w_leaf = 43, 45
    if output_3d_plot:
        n_leaf_components = len(leaf_pca.components)

        # mean_shape = leaf_pca.decode(np.zeros(n_leaf_components)).reshape(h_leaf, w_leaf, 3)

        for i in tqdm.tqdm(range(n_leaf_components)):

            _pc_all = []

            for index, t in enumerate(np.linspace(-max_range, max_range, n_frames)):
                data_modified = np.zeros(n_leaf_components)
                data_modified[i] += t
                curve_info_recover = leaf_pca.decode(data_modified)
                curve_info_recover = torch.from_numpy(curve_info_recover).float().cuda()

                # _pc = curve_info_recover.reshape(h_leaf, w_leaf, 3) + leaf_pca.data_mean.reshape(h_leaf, w_leaf, 3)
                _pc = curve_info_recover.reshape(h_leaf, w_leaf, 3).detach().cpu().numpy()
                # _diff = _pc - mean_shape

                components_viz_dir = f'/home/tianhang/data/folio/Folio Leaf Dataset/Folio/{species}_rotated_mask/leaf_deformation_pca_viz/{i}'
                os.makedirs(components_viz_dir, exist_ok=True)
                save_path = os.path.join(components_viz_dir, f'{str(index).zfill(6)}.png')

                mesh = grid_to_mesh(_pc)

                # paint color based on diff
                # _diff_normalized = (_diff - _diff.min()) / (_diff.max() - _diff.min())
                # colors = plt.get_cmap('jet')(_diff_normalized[..., 2])[:, :, :3]
                # mesh.vertex_colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
                # o3d.visualization.draw_geometries([mesh])

                save_path2 = os.path.join(f'/home/tianhang/code/Demeter-project.github.io/static/interpolation_3d_leaf/{species}/pca{i+1}', 
                                          f'{str(index).zfill(6)}.jpg')
                os.makedirs(os.path.dirname(save_path2), exist_ok=True)
                load_view_point(
                        [mesh],
                        # cam_json_path='/home/tianhang/code/Plant/cam_leaf_pca.json', 
                        cam_json_path='/home/tianhang/code/Plant/ScreenCamera_2025-09-23-22-35-11.json',
                        img_save_path=save_path2,
                        show_window=False,
                        show_normal_color=True,
                        show_normals=False,
                        normal_scale=0.05,
                        light_on=False
                    )

            _=1

    """
    visualize Leaf PCA components as 3D plt plot
    """
    output_3d_plot = False
    n_frames = 50
    max_range = 2.0
    h_leaf, w_leaf = 43, 45
    if output_3d_plot:
        n_leaf_components = len(leaf_pca.components)
        for i in tqdm.tqdm(range(n_leaf_components)):

            _pc_all = []

            for index, t in enumerate(np.linspace(-max_range, max_range, n_frames)):
                data_modified = np.zeros(n_leaf_components)
                data_modified[i] += t
                curve_info_recover = leaf_pca.decode(data_modified)
                curve_info_recover = torch.from_numpy(curve_info_recover).float().cuda()

                # _pc = curve_info_recover.reshape(h_leaf, w_leaf, 3) + leaf_pca.data_mean.reshape(h_leaf, w_leaf, 3)
                _pc = curve_info_recover.reshape(h_leaf, w_leaf, 3)
                _pc_all.append(_pc)
            
            _pc_all = torch.stack(_pc_all, dim=0)
            _pc_all_expand = torch.nn.functional.interpolate(_pc_all.permute(0, 3, 1, 2), size=(h_leaf+1, w_leaf+2), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            _pc_all_normal = grid_to_normals(_pc_all_expand)[:, :, 1:].detach().cpu().numpy()
            _pc_all = _pc_all.detach().cpu().numpy()

            for index, t in tqdm.tqdm(enumerate(np.linspace(-max_range, max_range, n_frames))):

                components_viz_dir = f'/home/tianhang/data/folio/Folio Leaf Dataset/Folio/{species}_rotated_mask/leaf_deformation_pca_viz/{i}'
                os.makedirs(components_viz_dir, exist_ok=True)

                save_path = os.path.join(components_viz_dir, f'{str(index).zfill(6)}.png')

                _pc = _pc_all[index]

                dpi = 200
                fig = plt.figure(figsize=(1200/dpi, 1200/dpi), dpi=dpi)
                ax = fig.add_subplot(111, projection='3d')
                
                # x_range = (-0.15, 0.65)
                # y_range = (-0.4, 0.4)
                # z_range = (-0.4, 0.4)
                # x_range = (-1.2, 1.2)
                # y_range = (-0.2, 2.2)
                # z_range = (-1.2, 1.2)
                x_range = (-0.5, 0.5)
                y_range = (-0.05, 0.95)
                z_range = (-0.5, 0.5)
                x_range_expand = np.ones([h_leaf, w_leaf]) * x_range[0]
                y_range_expand = np.ones([h_leaf, w_leaf]) * y_range[1]
                z_range_expand = np.ones([h_leaf, w_leaf]) * z_range[0]

                X, Y, Z = _pc[..., 0], _pc[..., 1], _pc[..., 2]

                _normal = _pc_all_normal[index]

                p0 = _pc[h_leaf//2, 0]
                p1 = _pc[h_leaf//2, -1]
                p2 = _pc[0, w_leaf//2]
                p3 = _pc[-1, w_leaf//2]

                p0_x = np.linspace(p0, np.array([x_range[0], p0[1], p0[2]]), 10)
                p0_y = np.linspace(p0, np.array([p0[0], y_range[1], p0[2]]), 10)
                p0_z = np.linspace(p0, np.array([p0[0], p0[1], z_range[0]]), 10)
                p1_x = np.linspace(p1, np.array([x_range[0], p1[1], p1[2]]), 10)
                p1_y = np.linspace(p1, np.array([p1[0], y_range[1], p1[2]]), 10)
                p1_z = np.linspace(p1, np.array([p1[0], p1[1], z_range[0]]), 10)
                p2_x = np.linspace(p2, np.array([x_range[0], p2[1], p2[2]]), 10)
                p2_y = np.linspace(p2, np.array([p2[0], y_range[1], p2[2]]), 10)
                p2_z = np.linspace(p2, np.array([p2[0], p2[1], z_range[0]]), 10)
                p3_x = np.linspace(p3, np.array([x_range[0], p3[1], p3[2]]), 10)
                p3_y = np.linspace(p3, np.array([p3[0], y_range[1], p3[2]]), 10)
                p3_z = np.linspace(p3, np.array([p3[0], p3[1], z_range[0]]), 10)

                elev=33
                azim=-60

                # caluculate viewing direction based on the view angle (elev, azim)
                # r = 1
                # theta = np.radians(elev)
                # phi = np.radians(azim)
                # view_dir = np.array([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])

                light_pos = np.array([-1,2,1]).astype(np.float32)
                light_vec = light_pos - _pc
                light_vec = light_vec / np.linalg.norm(light_vec, axis=-1, keepdims=True)
                # reflect_vec = 2 * np.sum(light_vec * _normal, axis=-1, keepdims=True) * _normal - light_vec
                # reflect_vec = reflect_vec / np.linalg.norm(reflect_vec, axis=-1, keepdims=True)
                diffuse_light = np.sum(light_vec * -_normal, axis=-1, keepdims=True)
                diffuse_light = np.clip(diffuse_light, 0, 1)
                # specular_light = 4 * np.sum(reflect_vec * view_dir, axis=-1, keepdims=True) ** 10

                light_intensity = diffuse_light 
                light_intensity = np.clip(light_intensity, 0, 1)

                ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, alpha=1.0, color='black', linewidth=0.3)
                ax.plot_wireframe(x_range_expand, Y, Z, rstride=1, cstride=1, color= red, alpha=0.5, linewidth=0.3)
                ax.plot_wireframe(X, y_range_expand, Z, rstride=1, cstride=1, color= green, alpha=0.5, linewidth=0.3)
                ax.plot_wireframe(X, Y, z_range_expand, rstride=1, cstride=1, color= blue, alpha=0.5, linewidth=0.3)

                # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolor='none', label='leaf surface', alpha=0.95, facecolors=np.ones_like(light_intensity) * np.array([1,1,1]))
                # ax.plot_surface(x_range_expand, Y, Z, rstride=1, cstride=1, edgecolor='none', facecolors=np.ones_like(Z)[..., None] * red, alpha=0.5, label='x-projection')
                # ax.plot_surface(x_range_expand, Y, Z, rstride=1, cstride=1, edgecolor='none', facecolors=np.ones_like(Z)[..., None] * red, alpha=0.5, label='x-projection')
                # ax.plot_surface(X, y_range_expand, Z, rstride=1, cstride=1, edgecolor='none', facecolors=np.ones_like(Z)[..., None] * green, alpha=0.5, label='y-projection')
                # ax.plot_surface(X, Y, z_range_expand, rstride=1, cstride=1, edgecolor='none', facecolors=np.ones_like(Z)[..., None] * blue, alpha=0.5, label='z-projection')
                # ax.scatter(p0[0], p0[1], p0[2], color=blue, s=10, marker='x')
                # ax.scatter(p1[0], p1[1], p1[2], color=blue, s=10, marker='x')
                # ax.scatter(p2[0], p2[1], p2[2], c=blue, s=10, marker='x')
                # ax.scatter(p3[0], p3[1], p3[2], c=blue, s=10, marker='x')
                # ax.plot(p0_x[:,0], p0_x[:,1], p0_x[:,2], c='gray', linewidth=1, ls='--')
                # ax.plot(p0_y[:,0], p0_y[:,1], p0_y[:,2], c='gray', linewidth=1, ls='--')
                # ax.plot(p0_z[:,0], p0_z[:,1], p0_z[:,2], c='gray', linewidth=1, ls='--')
                # ax.plot(p1_x[:,0], p1_x[:,1], p1_x[:,2], c='gray', linewidth=1, ls='--')
                # ax.plot(p1_y[:,0], p1_y[:,1], p1_y[:,2], c='gray', linewidth=1, ls='--')
                # ax.plot(p1_z[:,0], p1_z[:,1], p1_z[:,2], c='gray', linewidth=1, ls='--')
                # ax.plot(p2_x[:,0], p2_x[:,1], p2_x[:,2], c='gray', linewidth=1, ls='--')
                # ax.plot(p2_y[:,0], p2_y[:,1], p2_y[:,2], c='gray', linewidth=1, ls='--')
                # ax.plot(p2_z[:,0], p2_z[:,1], p2_z[:,2], c='gray', linewidth=1, ls='--')
                # ax.plot(p3_x[:,0], p3_x[:,1], p3_x[:,2], c='gray', linewidth=1, ls='--')
                # ax.plot(p3_y[:,0], p3_y[:,1], p3_y[:,2], c='gray', linewidth=1, ls='--')
                # ax.plot(p3_z[:,0], p3_z[:,1], p3_z[:,2], c='gray', linewidth=1, ls='--')

                # title = f'Component[{i}] = {t:.2f}'
                # ax.set_title(title)
                ax.set_xlim(x_range)
                ax.set_ylim(y_range)
                ax.set_zlim(z_range)
                # plt.legend()
                # ax.set_xlabel('X', fontweight='bold')
                # ax.set_ylabel('Y', fontweight='bold')
                # ax.set_zlabel('Z', fontweight='bold')
                # set view
                ax.view_init(elev=33, azim=-60)
                ax.patch.set_alpha(0)
                ax.grid(False)
                plt.tight_layout()
                # plt.show(block=True)

                # hide axis
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

                ax.grid(False)

                # hide axis panes (the "walls")
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False

                for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                    axis.pane.fill = False
                    axis.pane.set_edgecolor((0.5, 0.5, 0.5, 1))  # transparent edge
                    axis.line.set_color((0.5, 0.5, 0.5, 1))      # transparent axis line

                plt.savefig(save_path, dpi=dpi)
                print(f'Saved {save_path}')

                if species == 'ribes':
                    species_replace = 'papaya'
                else:
                    species_replace = species

                save_path2 = os.path.join(f'/home/tianhang/code/Demeter-project.github.io/static/interpolation_3d_leaf/{species_replace}/pca{i+1}', f'{str(index).zfill(6)}.jpg')
                os.makedirs(os.path.dirname(save_path2), exist_ok=True)
                plt.savefig(save_path2, dpi=dpi)

                # plt.show()
                    
                plt.close()

                if i>3:
                    break

            _=1

    # load stem data
    stem_3d = []
    for instance in instance_list:
        stem_folder = f'{folder}/{instance}/stem_template_deform'
        stem_paths = glob.glob(f'{stem_folder}/*.pth')
        print(f'Loading {len(stem_paths)} stem paths from {stem_folder}')
        for stem_path in stem_paths:
            stem_3d.append(torch.load(stem_path, weights_only=True))
    
    stem_3d = torch.stack(stem_3d, dim=0) # (n, 9, 3)
    b = stem_3d.shape[0]

    if species == 'rose':
        n_components = 10
    else:
        n_components = 0.995
    stem_pca = NodePCA(n_components=n_components)
    stem_pca.train(stem_3d.reshape(b, -1)) # (n, 9*3)
    stem_pca.save(f'/home/tianhang/data/{species}/3d_stem_pca.pth')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    viz = stem_3d.detach().cpu().numpy()
    for i in range(viz.shape[0]):
        ax.plot(viz[i, :, 0], viz[i, :, 1], viz[i, :, 2], linewidth=2.5)
    plt.show()
    
    output_3d_plot = False
    if output_3d_plot:
        n_stem_components = len(stem_pca.components)
        for i in tqdm.tqdm(range(n_stem_components)):

            components_viz_dir = f'/home/tianhang/data/folio/Folio Leaf Dataset/Folio/{species}_rotated_mask/stem_deformation_pca_viz/{i}'
            os.makedirs(components_viz_dir, exist_ok=True)

            for index, t in enumerate(np.linspace(-1, 1, n_frames)):
                data_modified = np.zeros(n_stem_components)
                data_modified[i] += t
                curve_info_recover = stem_pca.decode(data_modified)
                save_path = os.path.join(components_viz_dir, f'curve_{index}.png')

                # _pc = _decode(curve_info_recover.reshape(-1, 3))
                _pc = curve_info_recover.reshape(-1, 3) 
                # _pc_viz = curve.evaluate(num_points=2, w=1, cp=_pc, canonical=True)[1].detach().cpu().numpy()
                _pc_viz = _pc

                dpi = 200
                fig = plt.figure(figsize=(1200/dpi, 1200/dpi), dpi=dpi)
                ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(_pc_viz[:,0], _pc_viz[:,1], _pc_viz[:, 2], cmap='coolwarm', label='stem curve', s=8, c=len(_pc_viz)-np.arange(len(_pc_viz)))
                ax.plot(_pc_viz[:,0], _pc_viz[:,1], _pc_viz[:, 2],  label='stem curve', c='black', linewidth=3)
                x_range = (-0.6, 0.6)
                y_range = (-0.6, 0.6)
                z_range = (-0.1, 1.1)
                ax.set_xlim(x_range)
                ax.set_ylim(y_range)
                ax.set_zlim(z_range)
                x_range_expand = np.ones([len(_pc_viz)]) * x_range[0]
                y_range_expand = np.ones([len(_pc_viz)]) * y_range[1]
                z_range_expand = np.ones([len(_pc_viz)]) * z_range[0]
                
                _pc_proj_x_viz = np.stack([x_range_expand, _pc_viz[:,1], _pc_viz[:, 2]], axis=1)
                _pc_proj_y_viz = np.stack([_pc_viz[:,0], y_range_expand, _pc_viz[:, 2]], axis=1)
                _pc_proj_z_viz = np.stack([_pc_viz[:,0], _pc_viz[:,1], z_range_expand], axis=1)

                _p0 = _pc_viz[0]
                _p1 = _pc_viz[-1]

                _p0_x = np.linspace(_p0, _pc_proj_x_viz[0], 10)
                _p0_y = np.linspace(_p0, _pc_proj_y_viz[0], 10)
                _p0_z = np.linspace(_p0, _pc_proj_z_viz[0], 10)
                _p1_x = np.linspace(_p1, _pc_proj_x_viz[-1], 10)
                _p1_y = np.linspace(_p1, _pc_proj_y_viz[-1], 10)
                _p1_z = np.linspace(_p1, _pc_proj_z_viz[-1], 10)

                ax.plot(_pc_proj_x_viz[:, 0], _pc_proj_x_viz[:,1], _pc_proj_x_viz[:, 2], linewidth=2, label='x-projection', c=red, alpha=0.5)
                ax.plot(_pc_proj_y_viz[:, 0], _pc_proj_y_viz[:,1], _pc_proj_y_viz[:, 2], linewidth=2, label='y-projection', c=green, alpha=0.5)
                ax.plot(_pc_proj_z_viz[:, 0], _pc_proj_z_viz[:,1], _pc_proj_z_viz[:, 2], linewidth=2, label='z-projection', c=blue, alpha=0.5)

                ax.plot(_p0_x[:,0], _p0_x[:,1], _p0_x[:,2], c='gray', linewidth=1, ls='--')
                ax.plot(_p0_y[:,0], _p0_y[:,1], _p0_y[:,2], c='gray', linewidth=1, ls='--')
                ax.plot(_p0_z[:,0], _p0_z[:,1], _p0_z[:,2], c='gray', linewidth=1, ls='--')
                ax.plot(_p1_x[:,0], _p1_x[:,1], _p1_x[:,2], c='gray', linewidth=1, ls='--')
                ax.plot(_p1_y[:,0], _p1_y[:,1], _p1_y[:,2], c='gray', linewidth=1, ls='--')
                ax.plot(_p1_z[:,0], _p1_z[:,1], _p1_z[:,2], c='gray', linewidth=1, ls='--')


                ax.scatter(_p0[0], _p0[1], _p0[2], color=red, s=10, marker='x')
                ax.scatter(_p1[0], _p1[1], _p1[2], color=green, s=10, marker='x')

                # ax.set_xlabel('X', fontweight='bold')
                # ax.set_ylabel('Y', fontweight='bold')
                # ax.set_zlabel('Z', fontweight='bold')

                ax.patch.set_alpha(0)
                ax.grid(False)
                
                # hide axis
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

                ax.grid(False)

                # hide axis panes (the "walls")
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False

                for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                    axis.pane.fill = False
                    axis.pane.set_edgecolor((0.5, 0.5, 0.5, 1))  # transparent edge
                    axis.line.set_color((0.5, 0.5, 0.5, 1))      # transparent axis line

                # plt.legend()
                # title = f'Component[{i}] = {t:.2f}'
                # ax.set_title(title)
                ax.view_init(elev=33, azim=-52)
                # plt.show(block=True)
                plt.tight_layout()

                plt.savefig(save_path, dpi=dpi)
                print(f'Saved {save_path}')

                if species == 'ribes':
                    species_replace = 'papaya'
                else:
                    species_replace = species

                save_path2 = os.path.join(f'/home/tianhang/code/Demeter-project.github.io/static/interpolation_3d_stem/{species_replace}/pca{i+1}', f'{str(index).zfill(6)}.jpg')
                os.makedirs(os.path.dirname(save_path2), exist_ok=True)
                plt.savefig(save_path2, dpi=dpi)

                plt.close()

                if i>3:
                    break

    _ = 1

if __name__ == '__main__':

    # train_3d_pca('soybean')
    # train_3d_pca('ribes')
    # train_3d_pca('tobacco')
    # train_3d_pca('maize')
    train_3d_pca('rose')