# -*- coding: utf-8 -*-
import os
from scannetpp.common.scene_release import ScannetppScene_Release
from pathlib import Path
import json
import numpy as np
import torch
import cv2
from tqdm import tqdm
# from scannetpp.iphone.prepare_iphone_data import extract_rgb, extract_masks, extract_depth
import imageio.v3 as iio
import open3d as o3d
from scannetpp.common.utils.utils import run_command, load_yaml_munch, load_json, read_txt_list
from scannetpp.common.utils.dslr import compute_undistort_intrinsic
from scannetpp.common.utils.colmap import get_camera_images_poses, camera_to_intrinsic
from scannetpp.common.utils.anno import get_bboxes_2d, get_sem_ids_on_2d, get_visiblity_from_cache, get_vtx_prop_on_2d, load_anno_wrapper, viz_sem_ids_2d, get_single_image_visibility
from scannetpp.common.file_io import read_txt_list
from scannetpp.common.scene_release import ScannetppScene_Release
from scannetpp.common.utils.image import get_img_crop, load_image, save_img, viz_ids
from scannetpp.common.utils.rasterize import get_fisheye_cameras_batch, get_opencv_cameras_batch, prep_pt3d_inputs, rasterize_mesh
from pytorch3d.structures import Meshes

def save_json(data, file_path, indent=4, ensure_ascii=False):
    """
    �����ݱ���Ϊ JSON �ļ�
    :param data: Ҫ��������ݣ��ֵ���б���
    :param file_path: �ļ�·������ "data/output.json"��
    :param indent: �����ո�����������ʽ��None ��ʾ���մ洢��
    :param ensure_ascii: �Ƿ�ת���? ASCII �ַ���False �������ĵȣ�
    """
    file_dir = os.path.dirname(file_path)
    if file_dir:  
        os.makedirs(file_dir, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

def read_txt_to_list(file_path, keep_newlines=False, skip_empty=False):
    """
    �߼��棺�ɿ����Ƿ������з�����������

    ����:
        file_path (str): �ı��ļ�·��
        keep_newlines (bool): �Ƿ�����β���з���Ĭ��False��
        skip_empty (bool): �Ƿ��������У�Ĭ��False��

    ����:
        list: ��������������б�?
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        if keep_newlines:
            lines = [line for line in file]
        else:
            lines = [line.rstrip('\n') for line in file]
        
        if skip_empty:
            lines = [line for line in lines if line.strip() != '']
    
    return lines

class scannetpp_dataset:
    def __init__(self, data_dir, output_dir, output_json):
        self.data_dir = data_dir
        self.raw_data_dir = os.path.join(self.data_dir, 'data')
        self.existed_scene = os.listdir(self.raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_json = output_json
        self.val_scene_list_file = os.path.join(self.data_dir, 'splits/nvs_sem_val.txt')
        self.val_scene_list = read_txt_to_list(self.val_scene_list_file)
        self.sample_rate = 5 #ʵ�ʲ�����Ҫ*10
        self.image_type = 'iphone' # 'dslr'
        self.obj_visible_thresh = 0.1
        self.obj_pixel_thresh = 0.0001
        self.obj_dist_thresh = 999
        self.img_crop_dir = self.output_dir / 'img_crops'
        self.img_bbox_dir = self.output_dir / 'img_bbox'
        self.bbox_expand_factor = 0.1
        self.visiblity_cache_dir = os.path.join(output_dir, 'cache')
        self.undistort_dslr = True
        self.device = torch.device("cuda:0")
        self.unexpected_classes = ['wall', 'floor', 'ceiling', 'carpet', 'doorframe', 'split', 'SPLIT']

    def preprocess(self):
        print(len(self.val_scene_list))
        rasterout_dir = Path(self.data_dir) / self.image_type
        
        for scene_id in tqdm(self.val_scene_list):
            images_annotation = []
            print(f'processing {scene_id}')
            scene = ScannetppScene_Release(scene_id, data_root=Path(self.raw_data_dir))
            if not scene.scans_dir.exists() or not os.path.exists(os.path.join(self.raw_data_dir, scene_id, self.image_type)) or not scene.scan_anno_json_path.exists():
                continue
            anno = load_anno_wrapper(scene)
            # visibility_data = get_visiblity_from_cache(scene, rasterout_dir, self.visiblity_cache_dir, self.image_type, self.sample_rate, self.undistort_dslr, anno=anno)
            vtx_obj_ids = anno['vertex_obj_ids']
            # read mesh
            mesh = o3d.io.read_triangle_mesh(str(scene.scan_mesh_path))
            verts, faces, _ = prep_pt3d_inputs(mesh)
            obj_ids = np.unique(vtx_obj_ids)
            # remove 0
            obj_ids = sorted(obj_ids[obj_ids != 0])

            obj_id_locations = {obj_id: anno['objects'][obj_id]['obb']['centroid'] for obj_id in obj_ids}
            obj_id_size = {obj_id: anno['objects'][obj_id]['obb']['axesLengths'] for obj_id in obj_ids}
            obj_id_rotation = {obj_id: anno['objects'][obj_id]['obb']['normalizedAxes'] for obj_id in obj_ids}
            # obj_id_dims = {obj_id: anno['objects'][obj_id]['obb']['axesLengths'] for obj_id in obj_ids}

            # get the list of iphone/dslr images and poses
            # NOTE: should be the same as during rasterization
            colmap_camera, image_list, poses, distort_params = get_camera_images_poses(scene, self.sample_rate, self.image_type)
            # keep first 4 elements
            # distort_params = distort_params[:4]

            intrinsic = camera_to_intrinsic(colmap_camera)
            img_height, img_width = colmap_camera.height, colmap_camera.width

            # undistort_map1, undistort_map2 = None, None
            # go through list of images
            for i, image_name in enumerate(tqdm(image_list, desc='image', leave=False)):
                if self.image_type == 'iphone':
                    image_dir = scene.iphone_rgb_dir
                elif self.image_type == 'dslr':
                    image_dir = scene.dslr_resized_dir
                # load the image H, W, 3
                img_path = str(image_dir / image_name)
                img_relative_path = os.path.join(scene_id, self.image_type, image_name)
                if not Path(img_path).exists():
                    print(f'Image not found: {img_path}, skipping')
                    continue
                
                try:
                    print(f'Loading image: {img_path}')
                    img = load_image(img_path) 
                except:
                    print(f'Error loading image: {img_path}, skipping')
                    continue

                # rasterout_path = rasterout_dir / scene_id / f'{image_name}.pth'
                # raster_out_dict = torch.load(rasterout_path)
                # ��������raster����̫��������ռ�ݴ洢�ռ�̫�����԰�raster�Ĺ���ת�Ƶ����ʹ�û���
                pose = torch.Tensor(np.array(poses[i:i+1]))
                camera = get_opencv_cameras_batch(pose, img_height, img_width, intrinsic)
                mesh_verts = torch.Tensor(np.array([verts]))
                mesh_faces = torch.Tensor(np.array([faces]))
                mesh_torch = Meshes(verts=mesh_verts, faces=mesh_faces).to(self.device)
                raster_out_dict = rasterize_mesh(mesh_torch, img_height, img_width, camera)
                # raster_out_dict = self.get_raster_out()

                # visibility_data����������Ҳ̫ӷ���ˣ���Ҫ��дһ��
                # visibility_data = get_single_image_visibility(scene, rasterout_dir, self.visiblity_cache_dir, self.image_type, image_name, self.sample_rate, self.undistort_dslr, anno=anno)
                
                # if dimensions dont match, raster is from downsampled image
                # upsample using nearest neighbor
                pix_to_face = raster_out_dict['pix_to_face'].squeeze().cpu()
                # zbuf = raster_out_dict['zbuf'].squeeze().cpu()
                rasterized_dims = list(pix_to_face.shape)

                if rasterized_dims != [img_height, img_width]:
                    # upsample pixtoface and zbuf
                    pix_to_face = torch.nn.functional.interpolate(pix_to_face.unsqueeze(0).unsqueeze(0).float(), size=(img_height, img_width), mode='nearest').squeeze().squeeze().long()
                pix_to_face = pix_to_face.numpy()
                valid_pix_to_face =  pix_to_face[:, :] != -1
                # face_ndx = pix_to_face[valid_pix_to_face]

                # if undistort_map1 is not None and undistort_map2 is not None:
                #     # apply undistortion to rasterization (nearest neighbor), zbuf (linear) and image (linear)
                #     pix_to_face = cv2.remap(pix_to_face, undistort_map1, undistort_map2, 
                #         interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101,
                #     )
                #     # img is np
                #     img = cv2.remap(img, undistort_map1, undistort_map2,
                #         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                #     )
                # get object IDs on image
                try:
                    pix_obj_ids = get_vtx_prop_on_2d(pix_to_face, vtx_obj_ids, mesh)
                except IndexError: # something wrong with the rasterization
                    print(f'Rasterization error in {scene_id}/{image_name}, skipping')
                    continue

                # if cfg.dbg.viz_obj_ids: # save viz to file
                #     out_path = viz_obj_ids_dir / scene_id / f'{image_name}.png'
                #     viz_ids(img, pix_obj_ids, out_path)

                # # create semantics GT and of semantic ids on vertices, -1 = no semantic label
                # if cfg.save_semantic_gt_2d:
                #     out_path = save_dir / scene_id / f'{image_name}.png'
                #     if cfg.skip_existing_semantic_gt_2d and out_path.exists():
                #         print(f'File exists: {out_path}, skipping')
                #         continue
                #     # use 255 so that it can be saved as a PNG!
                #     pix_sem_ids = get_sem_ids_on_2d(pix_obj_ids, anno, semantic_classes, ignore_label=255)
                #     out_path.parent.mkdir(parents=True, exist_ok=True)
                #     # save to png file, smaller
                #     print(f'Saving 2d semantic anno to {out_path}')
                #     cv2.imwrite(str(out_path), pix_sem_ids)

                #     if cfg.viz_semantic_gt_2d:
                #         out_path = save_dir / scene_id / f'{image_name}_viz.png'
                #         print(f'Saving 2d semantic viz to {out_path}')
                #         viz_sem_ids_2d(pix_sem_ids, semantic_colors, out_path)
                #     continue # do only semantics, nothing else

                # get objid -> bbox x,y,w,h after upsampling rasterization, all the objs in this image
                bboxes_2d = get_bboxes_2d(pix_obj_ids)

                # go through each object that has a bbox 
                objs_info = []
                for obj_id, obj_bbox in bboxes_2d.items():
                    # ����
                    if obj_id == 0:
                        continue

                    # semantic label
                    obj_label = anno['objects'][obj_id]['label']
                    if obj_label in self.unexpected_classes:
                        continue

                    # ֻ���ݴ�С����
                    x, y, w, h = obj_bbox
                    if (w*h) / (img_height * img_width) < self.obj_pixel_thresh:
                        continue

                    # # faces in this image -> vertices in this image
                    # obj_mask_3d = vtx_obj_ids == obj_id
                    # obj_verts_ndx = np.where(obj_mask_3d)[0] # indices of vertices in this object
                    # faces_in_img = faces[face_ndx]
                    # img_verts = np.unique(faces_in_img)
                    # # obj verts in this image
                    # intersection = np.intersect1d(obj_verts_ndx, img_verts)
                    # # frac of obj vertices visible in this image
                    # visible_frac = len(intersection) / len(obj_verts_ndx)
                    # # enough of the object is seen
                    # if visible_frac < self.obj_visible_thresh:
                    #     continue

                    # obj_pixel_mask = pix_obj_ids == obj_id
                    # num_obj_pixels = np.sum(obj_pixel_mask)
                    # # check if obj occupies enough % of the image
                    # if num_obj_pixels / (img_height * img_width) < self.obj_pixel_thresh:
                    #     continue

                    # obj_zbuf = zbuf.squeeze()[obj_pixel_mask.squeeze()]
                    # # keep only the >= 0 values
                    # obj_zbuf = obj_zbuf[obj_zbuf >= 0]
                    # zbuf_min = obj_zbuf.min().item() if len(obj_zbuf) > 0 else -1
                    # if zbuf_min > self.obj_dist_thresh:
                    #     # object is too far away from camera
                    #     continue
                    
                    # if cfg.visibility_topk is not None:
                    #     images_visibilites = []
                    #     for i_name in visibility_data['images']:
                    #         if obj_id in visibility_data['images'][i_name]['objects'] and 'visible_vertices_frac' in visibility_data['images'][i_name]['objects'][obj_id]:
                    #             images_visibilites.append((i_name, visibility_data['images'][i_name]['objects'][obj_id]['visible_vertices_frac']))
                    #     # sort descending by visibility
                    #     images_visibilites.sort(key=lambda x: x[1], reverse=True)
                    #     top_images = [i_name for i_name, _ in images_visibilites][:cfg.visibility_topk]
                    #     # dont consider this object in this image
                    #     if image_name not in top_images: 
                    #         continue

                    # crop the object from the image
                    # img_crop = get_img_crop(img, obj_bbox, self.bbox_expand_factor, expand_bbox=True)
                    # img_crop_path = self.img_crop_dir / scene_id / f'{image_name}_{obj_id}.png'
                    # save_img(img_crop, img_crop_path)

                    # draw a bbox around the object and save the full image
                    # create new image with bbox of the object draw on the full image
                    # img_copy = img.copy()
                    # # x, y, w, h = obj_bbox
                    # # convert image RGB to BGR
                    # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
                    # cv2.rectangle(img_copy, (y, x), (y+h, x+w), (0, 0, 255), 2)
                    # # convert back to RGB
                    # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                    # global_prompt_img_path = self.output_dir / scene_id / f'{image_name}_{obj_id}_{obj_label}.png'
                    # # save it to file
                    # save_img(img_copy, global_prompt_img_path)
                    

                    # other useful info for this object
                    obj_location_3d = np.round(obj_id_locations[obj_id], 2).tolist()
                    # x, y, w, h = obj_bbox
                    # center of the bbox
                    # obj_location_2d = np.round([x + w/2, y + h/2]).tolist()
                    # obj_dims_3d = np.round(obj_id_dims[obj_id], 2).tolist()
                    
                    # vertices in this object
                    # obj_mask_3d = vtx_obj_ids == obj_id
                    obj_2Dbbox = np.round([x, y, x+w, y+h]).tolist()
                    objs_info.append({
                        "obj_id": str(obj_id),
                        "category": obj_label,
                        "3D_location": obj_location_3d,
                        "3D_size": obj_id_size[obj_id],
                        "3D_rotation": obj_id_rotation[obj_id],
                        "2D_bbox": obj_2Dbbox,
                    })

                # img_save_path = self.output_dir / scene_id / self.image_type / f'{image_name}.jpg'
                # save_img(img, img_save_path)
                images_annotation.append({
                    'scene_id': scene_id,
                    'image_name': image_name,
                    'image_path': img_relative_path,
                    'extrinsic': pose[0].cpu().numpy().tolist(),
                    'intrinsic': intrinsic.tolist(),
                    'objects' : objs_info
                })

            save_json(images_annotation, self.output_dir / scene_id / 'obj_annotation.json', indent=4, ensure_ascii=False)
                    
            


if __name__ == '__main__':
    data_dir = './'
    output_dir = './scannetpp_val_sampled1'
    output_json = './scannetpp_val_rawdata.json'
    dataset = scannetpp_dataset(data_dir=data_dir, output_dir=output_dir, output_json=output_json)
    dataset.preprocess()
