import os
import matplotlib.pyplot as plt
import tqdm
import imageio.v2 as iio
import numpy as np
import matplotlib.pyplot as plt
from utils.pca import NodePCA
import cv2
import tqdm

def bilinear_downsample(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """
    Downsample an image using bilinear interpolation.

    Args:
        img (np.ndarray): Input image of shape [h, w, 3].
        new_h (int): Desired height.
        new_w (int): Desired width.

    Returns:
        np.ndarray: Resized image of shape [new_h, new_w, 3].
    """
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def train_2d_pca(species: str):

    bound_type = 'circle'

    parent_folder = '/home/tianhang/data/folio/Folio Leaf Dataset/Folio'

    # image_foler = os.path.join(parent_folder, species)
    # keypoint_folder = os.path.join(parent_folder, species)
    rot_folder = os.path.join(parent_folder, species+'_rotated')
    os.makedirs(rot_folder, exist_ok=True)
    mask_folder = os.path.join(parent_folder, species+'_rotated_mask')
    os.makedirs(mask_folder, exist_ok=True)
    # curve_folder = image_foler

    # image_list = os.listdir(image_foler) 
    # image_list.sort()
    # image_list = [image_name for image_name in image_list if 'jpg' in image_name]
    # # image_list = image_list[split::total_splits]
    # if species == 'geranium':
    #     bad_sample = ['008', '012','016','017']
    #     image_list = [image_name for image_name in image_list if not any(bad in image_name for bad in bad_sample)]
 
    folder = f'/home/tianhang/data/folio/Folio Leaf Dataset/Folio/{species}_rotated_mask/skeleton/{bound_type}'
    data_files = os.listdir(folder)
    
    # rotate_90 = np.array([[0, -1], [1, 0]])
    rotate_90 = np.array([[0, 1], [-1, 0]])

    inners = []
    for data_file in data_files:
        data_path = os.path.join(folder, data_file)
        data = np.load(data_path)
        data_rotated = data @ rotate_90
        inners.append(data_rotated)
    h, w, _ = inners[0].shape
    inners = np.array(inners)
    # inners = np.array(inners).reshape(-1, h*w*2) # (n_instance, h*w*2)

    # fig = plt.figure()
    # plt.scatter(inners[:, :, 0, 0].reshape(-1), inners[:, :, 0, 1].reshape(-1), s=1)
    # plt.show(block=False)
    n_instance = inners.shape[0]

    pca = NodePCA(n_components=0.99)

    input_data = inners.reshape(n_instance, -1) # (n_instance, 2*n_points*2)
    data = pca.train(input_data) # (n_instance, n_components), the output is the PCA scores
    pca.save(os.path.join(mask_folder, 'skeleton_pca.pth'))

    sigma = np.std(data, axis=0)
    np.savetxt(os.path.join(mask_folder, 'skeleton_pca_sigma.txt'), sigma)

    mean_shape = pca.data_mean.detach().cpu().numpy()
    # mean_curve_reshape = mean_curve.reshape(2*n_points, 2)

    for i in tqdm.tqdm(range(len(pca.components))):
        print(f'Component {i}')

        components_viz_dir = f'/home/tianhang/data/folio/Folio Leaf Dataset/Folio/{species}_rotated_mask/skeleton_pca_viz/{i}'
        os.makedirs(components_viz_dir, exist_ok=True)

        component_scores = data[:, i]
        sigma = np.std(component_scores)
        # pca_range = (-sigma, sigma)


        # retexture
        # do_rexture = False
        # if do_rexture:

        #     num_points = 20

        #     shape_info_recover_sr = []
        #     for layer in shape_info_recover:
        #         layer = torch.from_numpy(layer).float().cuda()
        #         layer_sr = compute_catmull_rom_curve(wrap_control_points(layer), num_points=num_points).detach().cpu().numpy()
        #         shape_info_recover_sr.append(layer_sr)
        #     sr_num = shape_info_recover_sr[0].shape[0]
        #     shape_info_recover_sr[0] = np.ones([sr_num, 2]) * shape_info_recover[0][0:1] # set the first point to be the same as the original
        #     shape_info_recover_sr[-1] = np.ones([sr_num, 2]) * shape_info_recover[-1][0:1] # set the last point to be the same as the original
        #     shape_info_recover_sr = np.stack(shape_info_recover_sr, axis=0)

        #     shape_info_recover_sr2 = []
        #     for layer in shape_info_recover_sr.transpose(1, 0, 2):
        #         layer = torch.from_numpy(layer).float().cuda()
        #         layer_sr = compute_catmull_rom_curve(wrap_control_points(layer), num_points=num_points).detach().cpu().numpy()
        #         # plt.scatter(layer[:, 0].cpu().numpy(), layer[:, 1].cpu().numpy(), s=3, color='r')
        #         # plt.scatter(layer_sr[:, 0], layer_sr[:, 1], s=5, color='b')
        #         shape_info_recover_sr2.append(layer_sr)
        #     sr_num = shape_info_recover_sr2[0].shape[0]
        #     shape_info_recover_sr2 = np.stack(shape_info_recover_sr2, axis=0)
        #     shape_info_recover_sr2 = shape_info_recover_sr2.transpose(1, 0, 2)

        #     texture_path = os.path.join(parent_folder, species+'_rotated_mask', 'texture', 'square', f'0_1024.png')
        #     texture = iio.imread(texture_path)
        #     texture_on_template = texture.astype(np.float32) / 255.
        #     texture_on_template = np.flip(texture_on_template, axis=0) # flip to match the y-axis orientation

        #     texture_sampled = bilinear_downsample(texture_on_template, H, W)
        #     plt.scatter(shape_info_recover_sr2[:, :, 0], shape_info_recover_sr2[:, :, 1], color=texture_sampled.reshape(-1, 3), s=0.5)
        #     plt.show()

        #     exit()



        num_points = 2

        for index, t in enumerate(np.linspace(-2, 2, 50)):
            data_modified = np.zeros_like(data[0].copy()).astype(np.float32)
            data_modified[i] += t * sigma
            shape_info_recover = pca.decode(data_modified)

            shape_info_recover = shape_info_recover.reshape(h, w, 2)

            shape_info_recover_sr = shape_info_recover.copy()
            H, W, _ = shape_info_recover_sr.shape

            # Query texture
            # dpi = 300
            # fig = plt.figure(figsize=(1200/dpi, 1200/dpi), dpi=dpi)
            # texture_sampled = np.transpose(texture_sampled, (1, 0, 2)) # to match the x-y axis

            # viz_3D = False
            # if viz_3D:
                # ax = fig.add_subplot(111,projection='3d')
                # ax.set_title(f'Component {i}, t={t:.2f} sigma')
                # ax.plot_surface(shape_info_recover[:, :, 0], shape_info_recover[:, :, 1], np.zeros_like(shape_info_recover[:, :, 0]), alpha=1, color='g')
                # ax.view_init(elev=90, azim=-90, roll=0)
            # else:
                # ax = fig.add_subplot(111)
                # ax.set_title(f'Component {i}, t={t:.2f} sigma')
                # ax.plot(shape_info_recover[:, :, 0], shape_info_recover[:, :, 1], color='g', linewidth=1)
                # ax.scatter(shape_info_recover_sr[:, :, 0], shape_info_recover_sr[:, :, 1], color=texture_sampled.reshape(-1, 3), s=1)
            # plt.plot(shape_info_recover_sr2[:, :, 0], shape_info_recover_sr2[:, :, 1], color='g', linewidth=1)
            # plt.show()

            dpi = 200
            fig = plt.figure(figsize=(1200/dpi, 1200/dpi), dpi=dpi)
            # plt.scatter(shape_info_recover_sr[:, :, 0], shape_info_recover_sr[:, :, 1], color='g', s=2)
 
            ## plot green lines
            # horizontal lines
            for layer_id, layer in enumerate(shape_info_recover):
                plt.plot(layer[:, 0], layer[:, 1], c='black', linewidth=0.5, alpha=1)
            # vertical lines
            for layer_id, layer in enumerate(shape_info_recover.transpose(1, 0, 2)):
                if layer_id == 0:
                    c = 'blue'
                    linewidth = 1.0
                    plt.plot(layer[:, 0], layer[:, 1], c=c, linewidth=linewidth, alpha=1)
                elif layer_id == (W-1):
                    c = 'red'
                    linewidth = 1.0
                    plt.plot(layer[:, 0], layer[:, 1], c=c, linewidth=linewidth, alpha=1)
                else:
                    c = 'black'
                    linewidth = 0.3
                    plt.plot(layer[:, 0], layer[:, 1], c=c, linewidth=linewidth, alpha=0.8)

            # plt.show(block=False)
            # plt.xlim(-0.5, 0.5)
            # plt.ylim(-0.2, 0.8)
            if species == 'soybean':
                plt.xlim(-0.6, 0.6)
                plt.ylim(-0.15, 1.05)
            else:
                plt.xlim(-0.6, 0.6)
                plt.ylim(-0.3, 0.9)
            plt.axis('off')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.tight_layout()
            # plt.show()

            save_path = os.path.join(components_viz_dir, f'{str(index).zfill(6)}.jpg')
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            if i<=3:
                save_path2 = os.path.join(f'/home/tianhang/code/Demeter-project.github.io/static/interpolation/{species}/pca{i+1}', f'{str(index).zfill(6)}.jpg')
                fig.savefig(save_path2, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            _=1

            # do_viz_skeleton = True
            # if do_viz_skeleton:
            #     dpi = 300
            #     fig = plt.figure(figsize=(1200/dpi, 1200/dpi), dpi=dpi)

            #     pos_skeleton = shape_info_recover_sr.copy()
            #     n_joint = pos_skeleton.shape[1]

            #     for layer_id, layer in enumerate(pos_skeleton):
            #         plt.plot(layer[0:(n_joint)//2+1, 1], layer[:(n_joint)//2+1, 0], c='purple', linewidth=1.5, alpha=0.7)
            #         plt.plot(layer[(n_joint)//2:, 1], layer[(n_joint)//2:, 0], c='deeppink', linewidth=1.5, alpha=0.7)
            #         plt.scatter(layer[:, 1], layer[:, 0], s=9, facecolors='none', edgecolors='gray')
            #     # for layer_id, layer in enumerate(pos_skeleton.transpose(1, 0, 2)):
            #     #     if layer_id == (n_joint//2):
            #     #         continue
            #     #     plt.plot(layer[:, 1], layer[:, 0], c='orange', linewidth=0.8)
            #     plt.plot(pos_skeleton[:, (n_joint)//2, 1], pos_skeleton[:, n_joint//2, 0], c='orange', linewidth=1.5, alpha=0.7)
            #     # plt.scatter(u_joints, v_joints, s=5) 
            #     # plt.plot(l_seq_np[:, 1], l_seq_np[:, 0], c='r', linewidth=2)
            #     # plt.plot(r_seq_np[:, 1], r_seq_np[:, 0], c='b', linewidth=2)

            #     # if not canonical:
            #     l_p_viz = pos_skeleton[:, 0]
            #     r_p_viz = pos_skeleton[:, -1]
            #     plt.plot(l_p_viz[:, 1], l_p_viz[:, 0], c='r', linewidth=1.5, alpha=0.7)
            #     plt.plot(r_p_viz[:, 1], r_p_viz[:, 0], c='b', linewidth=1.5, alpha=0.7)

            #     # plt.xlim(-0.5, 0.5)
            #     # plt.ylim(-0.2, 1)
            #     # else:
            #     #     plt.plot(l_bound_value_leaf[:, 1], l_bound_value_leaf[:, 0], c='r', linewidth=2)
            #     #     plt.plot(r_bound_value_leaf[:, 1], r_bound_value_leaf[:, 0], c='b', linewidth=2)

            #     # plt.axis('off')
            #     # equal aspect ratio
            #     plt.gca().set_aspect('equal', adjustable='box')
            #     # plt.show(block=False)

            #     # plt.savefig(f'/home/tianhang/data/folio/Folio Leaf Dataset/Folio/{species}_rotated_mask/figure/l2t_solve_final.png', dpi=dpi, bbox_inches='tight', pad_inches=0)

            #     save_path = os.path.join(components_viz_dir, f'shape_{index}.png')
            #     fig.savefig(save_path, dpi=dpi)
            #     plt.close(fig)

            #     _=1

            # read the image, and center crop it to 80%
            # img = iio.imread(save_path)
            # _h, _w, _ = img.shape
            # center_h, center_w = _h//2, _w//2
            # new_h, new_w = int(_h*0.8), int(_w*0.8)
            # start_h, start_w = center_h - new_h//2, center_w - new_w//2
            # img = img[start_h:start_h+new_h, start_w:start_w+new_w]
            # iio.imwrite(save_path, img)

        _ = 1
    _ = 1

if __name__ == '__main__':
    
    # species = 'ashanti blood'
    # species = 'papaya'
    # species = 'geranium'
    # species = 'betel'
    # species = 'thevetia'
    # species = 'ficus'
    # species = 'soybean' # done
    # species = 'maize'
    # species = 'papaya' # done

    for species in ['geranium', 'soybean', 'papaya']:
        train_2d_pca(species)