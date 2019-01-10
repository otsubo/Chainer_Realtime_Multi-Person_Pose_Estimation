# -*- coding: utf-8 -*-
from enum import IntEnum

from models.CocoPoseNet import CocoPoseNet

#from models.FaceNet import FaceNet
#from models.HandNet import HandNet


class JointType(IntEnum):
    Rcollar = 0
    Lcollar = 1
    Rshoulder = 2
    Lshoulder = 3
    Rsleeve = 4
    Lsleeve = 5
    Rtrunk = 6
    Ctrunk = 7
    Ltrunk = 8
    Rhem = 9
    Chem = 10
    Lhem = 11

params = {
    'coco_dir': 'coco',
    'archs': {
        'posenet': CocoPoseNet,
    },
    # training params
    'min_keypoints': 5,
    'min_area': 32 * 32,
    'insize': 368,
    'downscale': 8,
    'paf_sigma': 8,
    'heatmap_sigma': 7,

    'min_box_size': 64,
    'max_box_size': 512,
    'min_scale': 0.5,
    'max_scale': 2.0,
    'max_rotate_degree': 40,
    'center_perterb_max': 40,

    # inference params
    'inference_img_size': 368,
    'inference_scales': [0.5, 1, 1.5, 2],
    # 'inference_scales': [1.0],
    'heatmap_size': 320,
    'gaussian_sigma': 2.5,
    'ksize': 17,
    'n_integ_points': 10,
    'n_integ_points_thresh': 8,
    'heatmap_peak_thresh': 0.05,
    'inner_product_thresh': 0.05,
    'limb_length_ratio': 1.0,
    'length_penalty_value': 1,
    'n_subset_limbs_thresh': 3,
    'subset_score_thresh': 0.2,
    'limbs_point': [
        [JointType.Rcollar, JointType.Rshoulder],
        [JointType.Rshoulder, JointType.Rsleeve],
        [JointType.Lcollar, JointType.Lshoulder],
        [JointType.Lshoulder, JointType.Lsleeve],
        [JointType.Rcollar, JointType.Ctrunk],
        [JointType.Lcollar, JointType.Ctrunk],
        [JointType.Rtrunk, JointType.Ctrunk],
        [JointType.Ltrunk, JointType.Ctrunk],
        [JointType.Rhem, JointType.Ctrunk],
        [JointType.Lhem, JointType.Ctrunk],
        [JointType.Chem, JointType.Ctrunk],
    ],
    'coco_joint_indices': [
        JointType.Rcollar,
        JointType.Lcollar,
        JointType.Rshoulder,
        JointType.Lshoulder,
        JointType.Rsleeve,
        JointType.Lsleeve,
        JointType.Rtrunk,
        JointType.Ctrunk,
        JointType.Ltrunk,
        JointType.Rhem,
        JointType.Chem,
        JointType.Lhem,
    ],
}
