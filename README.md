# Rethinking on Multi-Stage Networks for Human Pose Estimation

This repo is also linked to [github](https://github.com/megvii-detection/MSPN).

## This Version

This repo simply makes a training and testing these models easy by removing duplicate code and fixing things here and there. To run an example simply do:
```
python3 test.py --image https://media.npr.org/assets/img/2019/01/02/gettyimages-1058306908-0b38ff8a90d7bf88fea3133d8b72498665f63e12.jpg

----------------------------------------------------------------------
:: Loading the model
fetching https://media.npr.org/assets/img/2019/01/02/gettyimages-1058306908-0b38ff8a90d7bf88fea3133d8b72498665f63e12.jpg
:: out.size(): torch.Size([3, 256, 192])
:: Pass through model
:: Writing image at sample.png
----------------------------------------------------------------------
```

## Introduction
This is a pytorch realization of MSPN proposed in [ Rethinking on Multi-Stage Networks for Human Pose Estimation ][1]. In this work, we design an effective network MSPN to fulfill human pose estimation task.

Existing pose estimation approaches fall into two categories: single-stage and multi-stage methods. While multistage methods are seemingly more suited for the task, their performance in current practice is not as good as singlestage methods. This work studies this issue. We argue that the current multi-stage methods’ unsatisfactory performance comes from the insufficiency in various design choices. We propose several improvements, including the single-stage module design, cross stage feature aggregation, and coarse-tofine supervision. 

![Overview of MSPN.](/figures/MSPN.png)

The resulting method establishes the new state-of-the-art on both MS COCO and MPII Human Pose dataset, justifying the effectiveness of a multi-stage architecture.

### Model
Download ImageNet pretained ResNet-50 model from [Google Drive][6], and put it into **$MSPN_HOME/lib/models/**. For your convenience, We also provide a well-trained 2-stage MSPN model for COCO.


## Citation
Please considering citing our projects in your publications if they help your research.
```
@article{li2019rethinking,
  title={Rethinking on Multi-Stage Networks for Human Pose Estimation},
  author={Li, Wenbo and Wang, Zhicheng and Yin, Binyi and Peng, Qixiang and Du, Yuming and Xiao, Tianzi and Yu, Gang and Lu, Hongtao and Wei, Yichen and Sun, Jian},
  journal={arXiv preprint arXiv:1901.00148},
  year={2019}
}

@inproceedings{chen2018cascaded,
  title={Cascaded pyramid network for multi-person pose estimation},
  author={Chen, Yilun and Wang, Zhicheng and Peng, Yuxiang and Zhang, Zhiqiang and Yu, Gang and Sun, Jian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7103--7112},
  year={2018}
}
```
And the [code][7] of [Cascaded Pyramid Network][8] is also available. 

## Contact
You can contact us by email published in our [paper][1] or fenglinglwb@gmail.com.

[1]: https://arxiv.org/abs/1901.00148
[2]: https://pytorch.org/
[3]: https://github.com/cocodataset/cocoapi
[4]: http://cocodataset.org/#download
[5]: http://human-pose.mpi-inf.mpg.de/
[6]: https://drive.google.com/open?id=1MW27OY_4YetEZ4JiD4PltFGL_1-caECy
[7]: https://github.com/chenyilun95/tf-cpn
[8]: https://arxiv.org/abs/1711.07319
[9]: 
This repo is also linked to [github][9].

