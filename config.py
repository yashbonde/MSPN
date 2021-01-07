# encoding: utf-8
"""
@author: Wenbo Li
@contact: fenglinglwb@gmail.com
"""

import os, getpass
import os.path as osp
import argparse

from easydict import EasyDict as edict
# from dataset.attribute import load_dataset
# from cvpack.utils.pyt_utils import ensure_dir


class Config:
    # -------- Directoy Config -------- #
    USER = getpass.getuser()
    ROOT_DIR = osp.dirname(__file__)
    OUTPUT_DIR = osp.join(ROOT_DIR, 'model_logs', USER,
            osp.split(osp.split(osp.realpath(__file__))[0])[1])
    TEST_DIR = osp.join(OUTPUT_DIR, 'test_dir')
    TENSORBOARD_DIR = osp.join(OUTPUT_DIR, 'tb_dir') 

    # -------- Data Config -------- #
    DATALOADER = edict()
    DATALOADER.NUM_WORKERS = 4
    DATALOADER.ASPECT_RATIO_GROUPING = False
    DATALOADER.SIZE_DIVISIBILITY = 0

    DATASET = edict()
    DATASET.NAME = 'COCO'
    # dataset = load_dataset(DATASET.NAME)
    DATASET.KEYPOINT = {'NUM': 17,
                        'FLIP_PAIRS': [[1, 2],
                                       [3, 4],
                                       [5, 6],
                                       [7, 8],
                                       [9, 10],
                                       [11, 12],
                                       [13, 14],
                                       [15, 16]],
                        'UPPER_BODY_IDS': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        'LOWER_BODY_IDS': [11, 12, 13, 14, 15, 16],
                        'LOAD_MIN_NUM': 1}

    INPUT = edict()
    INPUT.NORMALIZE = True

    # edict will automatcally convert tuple to list, so ..
    INPUT.MEANS = [0.406, 0.456, 0.485]
    INPUT.STDS = [0.225, 0.224, 0.229]
    INPUT_SHAPE = (256, 192)
    OUTPUT_SHAPE = (64, 48)

    # -------- Model Config -------- #
    MODEL = edict()

    MODEL.BACKBONE = 'Res-50'
    MODEL.UPSAMPLE_CHANNEL_NUM = 256
    MODEL.STAGE_NUM = 2
    MODEL.OUTPUT_NUM = DATASET.KEYPOINT.NUM

    MODEL.DEVICE = 'cuda'

    MODEL.WEIGHT = osp.join(ROOT_DIR, 'lib/models/resnet-50_rename.pth')

    # -------- Training Config -------- #
    SOLVER = edict()
    SOLVER.BASE_LR = 5e-4 
    SOLVER.CHECKPOINT_PERIOD = 2400 
    SOLVER.GAMMA = 0.5
    SOLVER.IMS_PER_GPU = 32
    SOLVER.MAX_ITER = 96000 
    SOLVER.MOMENTUM = 0.9
    SOLVER.OPTIMIZER = 'Adam'
    SOLVER.WARMUP_FACTOR = 0.1
    SOLVER.WARMUP_ITERS = 2400 
    SOLVER.WARMUP_METHOD = 'linear'
    SOLVER.WEIGHT_DECAY = 1e-5 
    SOLVER.WEIGHT_DECAY_BIAS = 0

    LOSS = edict()
    LOSS.OHKM = True
    LOSS.TOPK = 8
    LOSS.COARSE_TO_FINE = True

    RUN_EFFICIENT = False 
    # -------- Test Config -------- #
    TEST = edict({'FLIP': True,
            'X_EXTENTION': 0.09,
            'Y_EXTENTION': 0.135,
            'SHIFT_RATIOS': [0.25],
            'GAUSSIAN_KERNEL': 5,
            'IMS_PER_GPU': 32})
    TEST.IMS_PER_GPU = 32


def get_config(dataset):
    config = Config()
    if dataset == "coco":
        config.DATASET.NAME = 'COCO'
        config.SOLVER.CHECKPOINT_PERIOD = 2400
        config.SOLVER.MAX_ITER = 96000
        config.RUN_EFFICIENT = False
    else:
        config.DATASET.NAME = 'MPII'
        config.SOLVER.CHECKPOINT_PERIOD = 1600
        config.SOLVER.MAX_ITER = 28800
        config.RUN_EFFICIENT = True
    return config


def link_log_dir(config):
    if not osp.exists('./log'):
        assert os.path.exists(config.OUTPUT_DIR)
        cmd = 'ln -s ' + config.OUTPUT_DIR + ' log'
        os.system(cmd)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', '--linklog', default=False, action='store_true')
    parser.add_argument('--dataset', choices = ["coco", "mpii"], default="coco", help = "which dataset to use")
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    config = get_config(args.dataset)
    if args.linklog:
        link_log_dir()
