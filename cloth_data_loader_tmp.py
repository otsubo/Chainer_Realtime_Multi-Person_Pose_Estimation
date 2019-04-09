# coding:utf-8
import os
import os.path as osp

import chainer
import json
import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split

import sys

import chainer
from chainer.dataset import DatasetMixin

import cv2
import math
import random

from cloth_entity_tmp import JointType, params


class ClothDataLoader(DatasetMixin):

    class_names = [
        '__backgrond__',
        'Rcollar',
        'Lcollar',
        'Rshoulder',
        'Lshoulder',
        'Rleeve',
        'Lleeve',
        'Rtrunk',
        'Ltrunl',
        'Rhem',
        'Lhem'
    ]
    mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))

    def __init__(self, split, return_image=False, img_aug=False):
        #assert mode in ['train', 'val', 'eval'], 'Data loading mode is invalid.'
        assert split in ('train', 'val')
        ids = self._get_ids()
        #print("ids_{}".format(ids))
        iter_train, iter_val = train_test_split(
            ids, test_size= 0.2, random_state=np.random.RandomState(1234))
        self.ids = iter_train if split == 'tarin' else iter_val
        self._return_image = return_image
        self._img_aug = img_aug

    def __len__(self):
        return len(self.ids)

    def _get_ids(self):
        ids = []
        dataset_dir = chainer.dataset.get_dataset_directory(
            'ClothV3')
        for data_id in os.listdir(dataset_dir):
            ids.append(osp.join('cloth_estimation',data_id))
            #ids.append(data_id)
        return ids

    def img_to_datum(self, img):
        img = img.copy()
        datum = img.astype(np.float32)
        datum = datum[:, :, ::-1] # RGB -> BGR
        datum -= self.mean_bgr
        datum = datum.transpose((2, 0, 1))
        datum = datum[np.newaxis,:,:,:]
        return datum

    def overlay_paf(self, img, paf):
        hue = ((np.arctan2(paf[1], paf[0]) / np.pi) / -2 + 0.5)
        saturation = np.sqrt(paf[0] ** 2 + paf[1] ** 2)
        saturation[saturation > 1.0] = 1.0
        value = saturation.copy()
        hsv_paf = np.vstack((hue[np.newaxis], saturation[np.newaxis], value[np.newaxis])).transpose(1, 2, 0)
        rgb_paf = cv2.cvtColor((hsv_paf * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        img = cv2.addWeighted(img, 0.6, rgb_paf, 0.4, 0)
        return img

    def overlay_pafs(self, img, pafs):
        mix_paf = np.zeros((2,) + img.shape[:-1])
        paf_flags = np.zeros(mix_paf.shape) # for constant paf
        for paf in pafs.reshape((int(pafs.shape[0]/2), 2,) + pafs.shape[1:]):
            paf_flags = paf != 0
            paf_flags += np.broadcast_to(paf_flags[0] | paf_flags[1], paf.shape)
            mix_paf += paf

        mix_paf[paf_flags > 0] /= paf_flags[paf_flags > 0]
        img = self.overlay_paf(img, mix_paf)
        return img

    def overlay_heatmap(self, img, heatmap):
        rgb_heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img = cv2.addWeighted(img, 0.6, rgb_heatmap, 0.4, 0)
        return img

    # return shape: (height, width)
    def generate_gaussian_heatmap(self, img_shape, joint, sigma):
        x, y = joint
        grid_x = np.tile(np.arange(img_shape[1]), (img_shape[0], 1))
        grid_y = np.tile(np.arange(img_shape[0]), (img_shape[1], 1)).transpose()
        grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
        gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
        return gaussian_heatmap

    def generate_heatmaps(self, img, pose, heatmap_sigma):
        heatmaps = np.zeros((0,) + img.shape[:-1])
        sum_heatmap = np.zeros(img.shape[:-1])
        for joint_index in range(len(JointType)):
            heatmap = np.zeros(img.shape[:-1])
            #for pose in poses:
            if pose[joint_index, 2] > 0:
                jointmap = self.generate_gaussian_heatmap(img.shape[:-1], pose[joint_index][:2], heatmap_sigma)
                heatmap[jointmap > heatmap] = jointmap[jointmap > heatmap]
                sum_heatmap[jointmap > sum_heatmap] = jointmap[jointmap > sum_heatmap]
            heatmaps = np.vstack((heatmaps, heatmap.reshape((1,) + heatmap.shape)))
        bg_heatmap = 1 - sum_heatmap  # background channel
        heatmaps = np.vstack((heatmaps, bg_heatmap[None]))
        return heatmaps.astype('f')

    # return shape: (2, height, width)
    def generate_constant_paf(self, img_shape, joint_from, joint_to, paf_width):
        if np.array_equal(joint_from, joint_to): # same joint
            return np.zeros((2,) + img_shape[:-1])

        joint_distance = np.linalg.norm(joint_to - joint_from)
        unit_vector = (joint_to - joint_from) / joint_distance
        rad = np.pi / 2
        rot_matrix = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        vertical_unit_vector = np.dot(rot_matrix, unit_vector)
        grid_x = np.tile(np.arange(img_shape[1]), (img_shape[0], 1))
        grid_y = np.tile(np.arange(img_shape[0]), (img_shape[1], 1)).transpose()
        horizontal_inner_product = unit_vector[0] * (grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1])
        horizontal_paf_flag = (0 <= horizontal_inner_product) & (horizontal_inner_product <= joint_distance)
        vertical_inner_product = vertical_unit_vector[0] * (grid_x - joint_from[0]) + vertical_unit_vector[1] * (grid_y - joint_from[1])
        vertical_paf_flag = np.abs(vertical_inner_product) <= paf_width
        paf_flag = horizontal_paf_flag & vertical_paf_flag
        constant_paf = np.stack((paf_flag, paf_flag)) * np.broadcast_to(unit_vector, img_shape[:-1] + (2,)).transpose(2, 0, 1)
        return constant_paf

    def generate_pafs(self, img, pose, paf_sigma):
        pafs = np.zeros((0,) + img.shape[:-1])

        for limb in params['limbs_point']:
            paf = np.zeros((2,) + img.shape[:-1])
            paf_flags = np.zeros(paf.shape) # for constant paf

            #for pose in poses:
            joint_from, joint_to = pose[limb]
            if joint_from[2] > 0 and joint_to[2] > 0:
                limb_paf = self.generate_constant_paf(img.shape, joint_from[:2], joint_to[:2], paf_sigma)
                limb_paf_flags = limb_paf != 0
                paf_flags += np.broadcast_to(limb_paf_flags[0] | limb_paf_flags[1], limb_paf.shape)
                paf += limb_paf

            paf[paf_flags > 0] /= paf_flags[paf_flags > 0]
            pafs = np.vstack((pafs, paf))
        return pafs.astype('f')

    #todo: 
    def generate_labels(self, img, pose):
        #img, ignore_mask, poses = self.augment_data(img, ignore_mask, poses)
        #resized_img, ignore_mask, resized_poses = self.resize_data(img, ignore_mask, poses, shape=(self.insize, self.insize))

        heatmaps = self.generate_heatmaps(img, pose, params['heatmap_sigma'])
        pafs = self.generate_pafs(img, pose, params['paf_sigma'])
        return img, pafs, heatmaps

    def shapes_to_pose(self, json_file):
        pose = []
        pose_tmp = []
        #json_file = osp.join(json_path, 'image.json')
        #pose = np.zeros((0, len(JointType), 3), dtype=np.int32)
        with open(json_file) as f:
            data = json.load(f)
        for shape in data['shapes']:
            pose_tmp = shape['points'][0]
            pose.append(pose_tmp)
        viz_vec = np.full((1, 10), 2)
        pose = np.insert(pose, 2, viz_vec, axis=1)
        return pose

    def json_file_to_lbl(self, img_shape, json_file):
        label_name_to_value = {}
        for label_value, label_name in enumerate(self.class_names):
            label_name_to_value[label_name] = label_value
        with open(json_file) as f:
            data = json.load(f)
        lbl = labelme.utils.shapes_to_label(
            img_shape, data['shapes'], label_name_to_value
        )
        return lbl

    def get_example(self, i):
        ann_id, data_id = self.ids[i].split('/')
        assert ann_id in ('cloth_estimation')
        dataset_dir = chainer.dataset.get_dataset_directory(
            'ClothV3')

        img_file = osp.join(dataset_dir,data_id, 'image.png')
        img = scipy.misc.imread(img_file)

        json_file = osp.join(dataset_dir, data_id, 'image.json')
        import ipdb; ipdb.set_trace()
        pose = self.shapes_to_pose(json_file)

        img, pafs, heatmaps = self.generate_labels(img, pose)
        img_datum = self.img_to_datum(img)
        img = np.array(img)
        img_datum = np.array(img_datum)
        if self._return_image:
            return img, pafs, heatmaps
        else:
            return img_datum, pafs, heatmaps

if __name__ =='__main__':
    import matplotlib.pyplot as plt
    dataset = ClothDataLoader('val', return_image=True, img_aug=False)
    for i in range(len(dataset)):
        img, pafs, heatmaps = dataset.get_example(i)
        img_to_show = img.copy()
        img_to_show = dataset.overlay_pafs(img_to_show, pafs)
        img_to_show = dataset.overlay_heatmap(img_to_show, heatmaps[:-1].max(axis=0))

        cv2.imshow('w', np.hstack((img, img_to_show)))
        k = cv2.waitKey(0)
        if k == ord('q'):
            sys.exit()
