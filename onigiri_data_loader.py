# coding:utf-8
import os
import os.path as osp

import imageio
import imgaug
import imgaug.augmenters as iaa

import chainer
import json
import numpy as np

from sklearn.model_selection import train_test_split

import labelme
import mvtk


import chainer
from chainer.dataset import DatasetMixin

import sys
for path in sys.path:
    if '/opt/ros/' in path:
        print('sys.path.remove({})'.format(path))
        sys.path.remove(path)
import cv2

import math
import random

from onigiri_entity import JointType, params


class OnigiriDataLoader(DatasetMixin):

    class_names = [
        '__backgrond__',
        'top',
        'left',
        'right'
    ]
    mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))

    def __init__(self, split, return_image=False, img_aug=False):
        #assert mode in ['train', 'val', 'eval'], 'Data loading mode is invalid.'
        assert split in ('train', 'val')
        ids = self._get_ids()
        ids = ids
        #print("ids_{}".format(ids))
        iter_train, iter_val = train_test_split(
            ids, test_size= 0.2, random_state=np.random.RandomState(1234))
        self.ids = iter_train if split == 'train' else iter_val
        self._return_image = return_image
        self._img_aug = img_aug

    def __len__(self):
        return len(self.ids)

    def _get_ids(self):
        ids = []
        dataset_dir = chainer.dataset.get_dataset_directory(
            '2019_11_28_pr2')
        for data_id in os.listdir(dataset_dir):
            ids.append(osp.join('5_onigiri_openpose',data_id))
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
        #mix_paf = np.zeros((2, 480, 640))
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
        with open(json_file) as f:
            data = json.load(f)
        if len(data['shapes']) == 10:
            for shape in data['shapes']:
                pose_tmp = shape['points'][0]
                pose.append(pose_tmp)
            viz_vec = np.full((1, 10), 2)
            pose = np.insert(pose, 2, viz_vec, axis=1)
            return pose
        else:
            joint_list = ['Rcollar', 'Lcollar', 'Rshoulder', 'Lshoulder', 'Rsleeve', 'Lsleeve', 'Rtrunk', 'Ltrunk', 'Rhem', 'Lhem']
            pose = np.zeros((10, 3))
            for shape in data['shapes']:
                index = joint_list.index(shape['label'])
                pose[index][0] = round(shape['points'][0][0])
                pose[index][1] = round(shape['points'][0][1])
                pose[index][2] = 2
            return pose

    def lbl_to_pose(self, lbl):
        pose = []
        pose_tmp = []
        for shape in data['shapes']:
            pose_tmp = shape['points'][0]
            pose.append(pose_tmp)
        viz_vec = np.full((1, 10), 2)
        pose = np.insert(pose, 2, viz_vec, axis=1)
        return pose

    def points_to_label(self, img_shape, shapes, label_name_to_value):
        cls = np.zeros((480, 640), dtype=np.int32)
        for shape in shapes:
            points = shape['points']
            label = shape['label']
            shape_type = shape.get('shape_type', None)
            cls_name = label
            cls_id = label_name_to_value[cls_name]
            mask = labelme.utils.shape_to_mask(img_shape[:2], points, 'point', line_width=1, point_size=1)
            cls[mask] = cls_id
        return cls

    def json_file_to_lbl(self, img_shape, json_file):
        label_name_to_value = {}
        for label_value, label_name in enumerate(self.class_names):
            label_name_to_value[label_name] = label_value
        with open(json_file) as f:
            data = json.load(f)
        lbl = self.points_to_label(img_shape, data['shapes'], label_name_to_value)

        return lbl

    def get_example(self, i):
        ann_id, data_id = self.ids[i].split('/')
        assert ann_id in ('5_onigiri_openpose')
        dataset_dir = chainer.dataset.get_dataset_directory(
            '')

        img_file = osp.join(dataset_dir,data_id, 'image.png')
        #print("img_file:{}".format(img_file))
        img = imageio.imread(img_file)
        img_shape = img.shape
        json_file = osp.join(dataset_dir, data_id, 'image.json')
        if self._img_aug:
            lbl = self.json_file_to_lbl(img_shape, json_file)
            obj_datum = dict(img=img)
            random_state = np.random.RandomState()
            st = lambda x: iaa.Sometimes(0.3 ,x)
            augs = [
                st(iaa.InColorspace(
                    'HSV', children=iaa.WithChannels([1, 2],
                                                     iaa.Multiply([0.3, 2.5])))),
                st(iaa.GaussianBlur(sigma=[0.0, 1.0])),
                st(iaa.AdditiveGaussianNoise(
                    scale=(0.0, 0.1 * 255), per_channel=True)),
            ]
            obj_datum = next(mvtk.aug.augment_object_data(
                [obj_datum], random_state=random_state, augmentations=augs))
            img = obj_datum['img']
            obj_datum2 = dict(img=img, lbl=lbl)
            random_state2 = np.random.RandomState()
            st2 = lambda x: iaa.Sometimes(0.7, x)  # NOQA
            augs2 = [
                st2(iaa.Affine(scale=(0.7, 1.3), order=0)),
                st2(iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})),
                st2(iaa.Affine(rotate=(-180, 180), order=0)),
                st2(iaa.Affine(shear=(-30, 30), order=0)),
            ]
            obj_datum2 = next(mvtk.aug.augment_object_data(
                [obj_datum2], random_state=random_state2, augmentations=augs2))
            img = obj_datum2['img']
            lbl = obj_datum2['lbl']
            pose = []
            pose_tmp = []
            for i in range(11):
                x = np.nanmean(np.where(lbl==i)[1]).round().astype('i')
                y = np.nanmean(np.where(lbl==i)[0]).round().astype('i')
                if x < 0 or y < 0:
                    pose.append((0, 0, 0))
                else:
                    pose.append((x, y, 2))
            pose = np.array(pose[1:])
        else:
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
    dataset = OnigiriDataLoader('train', return_image=True, img_aug=True)
    for i in range(len(dataset)):
        img, pafs, heatmaps = dataset.get_example(i)
        img_to_show = img.copy()
        img_to_show = dataset.overlay_pafs(img_to_show, pafs)
        img_to_show = dataset.overlay_heatmap(img_to_show, heatmaps[:-1].max(axis=0))

        cv2.imshow('w', np.hstack((img, img_to_show)))
        k = cv2.waitKey(0)
        if k == ord('q'):
            sys.exit()
