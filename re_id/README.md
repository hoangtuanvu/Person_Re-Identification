# Part-based Convolutional Baseline for Person Retrieval and the Refined Part Pooling

Code for the paper [Beyond Part Models: Person Retrieval with Refined Part Pooling (and A Strong Convolutional Baseline)](https://arxiv.org/pdf/1711.09349.pdf). 


## Preparation
<font face="Times New Roman" size=4>

**Prerequisite: Python 3 and Pytorch 0.4+**

1. Install [Pytorch](https://pytorch.org/)

2. Download dataset
	a. Market-1501 [BaiduYun](https://pan.baidu.com/s/1ntIi2Op?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=)
	b. Move them to ```~/data/market1501```
</font>

## train
<font face="Times New Roman" size=4>

```sh train.sh```
With Pytorch 0.4.0, we shall get about 92.5% rank-1 accuracy and 78.0% mAP on Market-1501.
</font>

## inference
<font face="Times New Roman" size=4>

```sh inference.sh```
Please put query images and galerries into sample/query and sample/gallery.
</font>

## Citiaion
<font face="times new roman" size=4>

Please cite this paper in your publications if it helps your research:
</font>

```
@inproceedings{sun2018PCB,
  author    = {Yifan Sun and
               Liang Zheng and
               Yi Yang and
			   Qi Tian and
               Shengjin Wang},
  title     = {Beyond Part Models: Person Retrieval with Refined Part Pooling (and A Strong Convolutional Baseline)},
  booktitle   = {ECCV},
  year      = {2018},
}
```
