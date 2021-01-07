"""
@author: Wenbo Li
@contact: fenglinglwb@gmail.com

This version: @yashbonde
"""
import os
import io
import cv2
import hashlib
import tempfile
import requests
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from network import MSPN
from config import get_config

# define the global config for now
cfg = None

def fetch(url):
    # efficient loading of URLS
    fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp) and os.stat(fp).st_size > 0:
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        print("fetching", url)
        dat = requests.get(url).content
        with open(fp+".tmp", "wb") as f:
            f.write(dat)
        os.rename(fp+".tmp", fp)
    return dat

def get_image_from_url(url):
    return Image.open(io.BytesIO(fetch(url)))


def get_preds_and_scores(outputs):
    kernel=11
    shifts=[0.25]

    nr_img = outputs.shape[0]
    preds = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 2))
    maxvals = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 1))
    for i in range(nr_img):
        score_map = outputs[i].copy()
        score_map = score_map / 255 + 0.5
        kps = np.zeros((cfg.DATASET.KEYPOINT.NUM, 2))
        scores = np.zeros((cfg.DATASET.KEYPOINT.NUM, 1))
        border = 10
        dr = np.zeros((cfg.DATASET.KEYPOINT.NUM,
            cfg.OUTPUT_SHAPE[0] + 2 * border, cfg.OUTPUT_SHAPE[1] + 2 * border))
        dr[:, border: -border, border: -border] = outputs[i].copy()
        for w in range(cfg.DATASET.KEYPOINT.NUM):
            dr[w] = cv2.GaussianBlur(dr[w], (kernel, kernel), 0)
        for w in range(cfg.DATASET.KEYPOINT.NUM):
            for j in range(len(shifts)):
                if j == 0:
                    lb = dr[w].argmax()
                    y, x = np.unravel_index(lb, dr[w].shape)
                    dr[w, y, x] = 0
                    x -= border
                    y -= border
                lb = dr[w].argmax()
                py, px = np.unravel_index(lb, dr[w].shape)
                dr[w, py, px] = 0
                px -= border + x
                py -= border + y
                ln = (px ** 2 + py ** 2) ** 0.5
                if ln > 1e-3:
                    x += shifts[j] * px / ln
                    y += shifts[j] * py / ln
            x = max(0, min(x, cfg.OUTPUT_SHAPE[1] - 1))
            y = max(0, min(y, cfg.OUTPUT_SHAPE[0] - 1))
            kps[w] = np.array([x * 4 + 2, y * 4 + 2])
            scores[w, 0] = score_map[w, int(round(y) + 1e-9), int(round(x) + 1e-9)]
            
            preds[i] = kps
            maxvals[i] = scores

    return preds, maxvals


def visualize(img, joints):
    # here the pairs are hardcoded representation of the model output.
    # the model predicts a list of values like:
    # array([[[126.,  71.],
    #         [127.,  70.],
    #         [125.,  70.],
    #         [133.,  70.],
    #         [125.,  74.],
    #         [138.,  77.],
    #         [123.,  90.],
    #         [141.,  90.],
    #         [131., 106.],
    #         [122.,  95.],
    #         [129., 106.],
    #         [154., 133.],
    #         [142., 137.],
    #         [153., 166.],
    #         [125., 174.],
    #         [159., 206.],
    #         [134., 209.]]])
    # 
    # but now we need to make sense of this list of prediction
    # so we then tell that for this list of points according to the COCO
    # dataset, each line corresponds to this different part of the human
    # pose.
    # These values come directly from the paper's authors though I believe
    # this actually comes directly from the dataset creators (COCO) and
    # author's for this paper simply used this value.
    #
    # This function takes in the input image and this set of predicted lines
    # and creates a mask for it. It is also used for creating the images for
    # readme.
    pairs = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
             [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
             [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    color = np.random.randint(0, 256, (len(pairs), 3)).tolist()

    for i in range(2):
        if joints[i, 0] > 0 and joints[i, 1] > 0:
            cv2.circle(img, tuple(joints[i, :2]), 2, tuple(color[i]), 2)

    def draw_line(img, p1, p2, c):
        if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
            cv2.line(img, tuple(p1), tuple(p2), c, 2)

    for i, pair in enumerate(pairs):
        draw_line(img, joints[pair[0] - 1], joints[pair[1] - 1], color[i])

    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default = "https://miro.medium.com/max/1200/1*56MtNM2fh_mdG3iGnD7_ZQ.jpeg", help="pass any image URL")
    parser.add_argument('--model_path', default = "drive-download-20210107T043037Z-001/mspn_2xstg_coco.pth")
    args = parser.parse_args()

    print("-"*70)
    print(":: Loading the model")
    cfg = get_config("coco")
    model = MSPN(cfg)
    state_dict = torch.load("drive-download-20210107T043037Z-001/mspn_2xstg_coco.pth", map_location="cpu")
    state_dict = state_dict['model']
    model.load_state_dict(state_dict)

    # define the image transformations to apply to each image
    image_transformations = transforms.Compose([
        transforms.Resize((256, 192)),                         # resize to a (256,192) image
        transforms.ToTensor(),                                 # convert to tensor
        transforms.Normalize(cfg.INPUT.MEANS, cfg.INPUT.STDS), # normalise image according to imagenet valuess
    ])

    test_image = get_image_from_url(args.image)
    out = image_transformations(test_image)
    print(":: out.size():", out.size())
    print(":: Pass through model")
    outputs = model(out.view(1, *out.size()))
    outputs = outputs.detach().numpy()

    preds, maxvals = get_preds_and_scores(outputs)

    resize = test_image.resize((192, 256))
    cv2img = cv2.cvtColor(np.array(resize), cv2.COLOR_RGB2BGR)
    cv2img = cv2.cvtColor(cv2img, cv2.COLOR_RGB2BGR)
    out_img = visualize(cv2img, preds[0].astype(int))
    out_img = cv2.resize(out_img, (256, 192))

    print(":: Writing image at sample.png")
    cv2.imwrite("./sample.png", out_img)
    print("-"*70)

