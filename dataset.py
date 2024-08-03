import logging

import numpy
import torch.utils.data as data
import torch

#from scipy.ndimage import imread
import imageio
import os
import os.path
import glob
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

from torchvision import transforms


def make_dataset(root, train):
  dataset = []
  if train:
    dir = os.path.join(root, 'train')
  else:
    dir = os.path.join(root, 'test')
  for fGT in glob.glob(os.path.join(dir, '*_mask.tif')):
    fName = os.path.basename(fGT)  # 获取所有*_mask.tif（真值y）文件的文件名
    fImg = fName[:-9] + '.tif'  # 获取对应.tif（数据x）的文件名
    dataset.append([os.path.join(dir, fImg), os.path.join(dir, fName)])

  return dataset


#使用原文数据集分割
def make_seg_dataset(root, train):
  dataset1 = []
  if train:
    dir = os.path.join(root, 'train')
  else:
    dir = os.path.join(root, 'test')
  for fGT1 in glob.glob(os.path.join(dir, '*_mask.jpg')):
    fName = os.path.basename(fGT1)  # 获取所有*_mask.png（真值y）文件的文件名
    fImg = fName[:-9] + '.jpg'  # 获取对应.png（数据x）的文件名
    #logging.info(fImg)
    dataset1.append([os.path.join(dir, fImg), os.path.join(dir, fName)])

  return dataset1



class kaggle2016nerve(data.Dataset):
  """
  Read dataset of kaggle ultrasound nerve segmentation dataset
  https://www.kaggle.com/c/ultrasound-nerve-segmentation
  """

  def __init__(self, root, train, transform=None):
    # 属性
    self.train = train

    # we cropped the image对图像进行裁剪
    self.nRow = 400
    self.nCol = 560
    #图片大小是560*400

    # if self.train:
    self.dataset_path = make_dataset(root, train)  # 由图像路径及其分割影像路径对（列表）组成的列表

  # idx的取值范围根据__len__的返回值确定，范围为0——len-1
  def __getitem__(self, idx):  # idx是指[[图1路径，分割1路径]，[图2路径，分割2路径],......[]]中每个元素[图n路径，分割n路径]的索引
    # if self.train:
    img_path, gt_path = self.dataset_path[idx]

    # 原始图像
    img = imageio.imread(img_path)
    # 网上说的：返回numpy，RGB图像，三通道，[H，W，C]。conv2d给定卷积核大小，in_channel即为此卷积层卷积核的通道数，out_channel为卷积核个数
    # 但跑了一下发现imageio.imread没有通道数啊？？_________________________________
    img = img[0:self.nRow, 0:self.nCol]# 显示图片的一部分
    #*************将不显示通道的情况，变为1通道*******************************
    img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
    #*******************************************************************
    # imageio.imread(img_path).size()是[400,560]
    # 因为np.atleast_3d将其转为了三维数组[400,560,1]，所以transpose(2, 0, 1)转换为[1，400，560]
    # 因此经过此语句，这里的大小为[1，400，560]即通道数为1
    #所以还是以一通道的图像输入网络中进行训练
    # 由于imageio.v2.imread读入的顺序是[H，W，C]，将其转换为[C，H，W]
    img = (img - img.min()) / (img.max() - img.min())
    img = torch.from_numpy(img).float()
    # 将numpy的array变为tensor，在GPU中进行训练

    # 分割图像
    gt = imageio.imread(gt_path)[0:self.nRow, 0:self.nCol]
    gt = np.atleast_3d(gt).transpose(2, 0, 1)
    gt = gt / 255.0
    # gt图像为二值的，给他归一化
    gt = torch.from_numpy(gt).float()

    return img, gt

  def __len__(self):
    return (len(self.dataset_path))



class segmentation_dataset(data.Dataset):
  def __init__(self, root, train, transform=None):
    self.train = train
    self.dataset_path1 = make_seg_dataset(root, train)  # 由图像路径及其分割影像路径对（列表）组成的列表

  # idx的取值范围根据__len__的返回值确定，范围为0——len-1
  def __getitem__(self, index):  # idx是指[[图1路径，分割1路径]，[图2路径，分割2路径],......[]]中每个元素[图n路径，分割n路径]的索引
      img_path, gt_path = self.dataset_path1[index]
      # 原始图像
      img = Image.open(img_path).convert("L")
      img = img.resize((256, 256), Image.ANTIALIAS)
      img = np.array(img)
      # *************将不显示通道的情况，变为1通道*******************************
      img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
      # *******************************************************************
      # imageio.imread(img_path).size()是[400,560]
      # 因为np.atleast_3d将其转为了三维数组[400,560,1]，所以transpose(2, 0, 1)转换为[1，400，560]
      # 因此经过此语句，这里的大小为[1，400，560]即通道数为1
      # 所以还是以一通道的图像输入网络中进行训练
      # 由于imageio.v2.imread读入的顺序是[H，W，C]，将其转换为[C，H，W]
      img = (img - img.min()) / (img.max() - img.min())
      img = torch.from_numpy(img).float()
      # print(img.size())
      # 将numpy的array变为tensor，在GPU中进行训练

      # 分割图像
      gt = Image.open(gt_path).convert("L")
      gt = gt.resize((256, 256), Image.ANTIALIAS)
      gt = np.array(gt)
      # 不是gt = gt > 0.01
      gt = np.atleast_3d(gt).transpose(2, 0, 1).astype(np.float32)
      gt = np.where(gt > 0, 1.0, 0.0)
      # gt = gt / 255.0
      # 归一化
      gt = torch.from_numpy(gt).float()
      # print('fanhuiqian:', gt)
      # print(gt.size())
      # gt = torch.where(gt > 0.01, torch.ones_like(gt), torch.zeros_like(gt))

      return img, gt

  def __len__(self):
    return (len(self.dataset_path1))



#图像分类数据处理
#一组数据由：一张图片+标签，组成
class classification_dataset(data.Dataset):
  def __init__(self, root, train, transform=None):
    if train:
      lu_jing = os.path.join(root, 'train')
    else:
      lu_jing = os.path.join(root, 'test')
    self.lu_jing = lu_jing#数据集路径
    self.shuju = os.listdir(self.lu_jing)# 展示出该路径下的所有数据名即图片名
    self.transform = transform# 是否要对图像进行预处理变换
    self.len = len(self.shuju)# 共有多少张图片

  def __getitem__(self, index):

      image_index = self.shuju[index]  # 获取每张图像的名字
      img_path = os.path.join(self.lu_jing, image_index)  # 获取每张图像的路径
      img = Image.open(img_path).convert("L")
      # 用Image.open打开的图片是PIL类型，默认RGB，它自带resize函数。
      # 需要输入的是一通道，因此将默认的RGB三通道变为灰度图L
      # Image.open打开后的PIL类型可以直接图片展示，但是不能直接读取其中的像素点值，且对其.size时只有宽和高
      # 只有将PIL类型转化为numpy类型：im = numpy.array(img)
      # 才能看到shape属性，是（height, width, channel）数组
      # 由于pytorch需要的的顺序是(batch,c,h,w)，所以需要使用transpose（2，0，1）将[h,w,c]转换为[c,h,w]
      img = img.resize((256, 256), Image.ANTIALIAS)
      # 将图像通过压缩展开，resize成固定大小
      # 转化为numpy类型,size为[h,w,c]
      img = np.array(img)
      # np.array(Image.open(img_path))也没有通道数，只有[400,560]
      img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)  # *******************************************
      # 经过此语句，img.shape为[1,400,560]
      '''
      if self.transform:
        img = self.transform(img)
      '''
      # 将numpy转换为tensor,在GPU中计算
      img = (img - img.min()) / (img.max() - img.min())
      img = torch.from_numpy(img).float()

      label = int(image_index[0])  # 图片名第一位存的是标签
      # label = self.oneHot(label)
      # 想使用交叉熵损失函数，则不能适应one-hot作为label编码
      # 直接输入代表类别的数字，比如0，1，2，3，4....，在使用交叉熵计算损失时会自动变为one-hot格式
      # print(img.size(), label)

      return img, label

  def __len__(self):
    return self.len

  # 将标签转为onehot编码
  # 本实验编码包括：non-COVID:10，COVID:01
  def oneHot(self, label):
    tem = np.zeros(2)
    tem[label] = 1
    return torch.from_numpy(tem)

# 每个客户端的dataloader
class DatasetSplit(Dataset):
  def __init__(self, dataset, idxs):
    self.dataset = dataset
    self.idxs = list(idxs)

  def __len__(self):
    return len(self.idxs)

  def __getitem__(self, item):
    image, gt = self.dataset[self.idxs[item]]
    return image, gt