import cv2
import imageio
import math
import os
import torch

import elpips
import lpips
import pytorch_ssim
from DISTS_pytorch import DISTS

from IPython import embed
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import csv #调用数据保存文件
import pandas as pd #用于数据输出

def lpipsmetric(img1, img2):
    use_gpu = False         # Whether to use GPU
    spatial = True         # Return a spatial map of perceptual distance.
    # Linearly calibrated models (LPIPS)
    loss_fn = lpips.LPIPS(net='alex', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
    # loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'
    if(use_gpu):
        loss_fn.cuda()
    ## Example usage with images
    image1 = lpips.im2tensor(lpips.load_image(path1))
    image2 = lpips.im2tensor(lpips.load_image(path2))
    if(use_gpu):
        image1 = image1.cuda()
        image2 = image2.cuda()
    d0 = loss_fn.forward(image1,image2)
    d1 = 1-d0
    d2 = 1-d0.mean()
    if not spatial:
        return d1
    else:
        return d2             # The mean distance is approximately the same as the non-spatial distance

def ssimmetric(img1, img2):
    image1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0)/255.0
    image2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0)/255.0
    if torch.cuda.is_available():
        image1 = image1.cuda()
        image2 = image2.cuda()
        # Functional: pytorch_ssim.ssim(image1, image2, window_size = 11, size_average = True)
        ssim_value = pytorch_ssim.ssim(image1, image2).item()
        return ssim_value

def distsmetric(img1, img2):
    D = DISTS()
    # calculate DISTS between X, Y (a batch of RGB images, data range: 0~1)
    # X: (N,C,H,W)
    # Y: (N,C,H,W)
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)
    image1 = torch.from_numpy(np.rollaxis(image1, 2)).float().unsqueeze(0)/255.0
    image2 = torch.from_numpy(np.rollaxis(image2, 2)).float().unsqueeze(0)/255.0
    dists_value = D(image1, image2)
    # set 'require_grad=True, batch_average=True' to get a scalar value as loss.
    dists_loss = D(image1, image2, require_grad=True, batch_average=True)
    dists_loss.backward()
    return dists_loss

def siftmetric(img1, img2):
    def getMatchNum(matches,ratio):
        '''返回特征点匹配数量和匹配掩码'''
        matchesMask=[[0,0] for i in range(len(matches))]
        matchNum=0
        for i,(m,n) in enumerate(matches):
            if m.distance<ratio*n.distance: #将距离比率小于ratio的匹配点删选出来
                matchesMask[i]=[1,0]
                matchNum+=1
        return (matchNum,matchesMask)
    comparisonImageList=[] #记录比较结果
    #创建SIFT特征提取器
    sift = cv2.xfeatures2d.SIFT_create()
    #创建FLANN匹配对象
    FLANN_INDEX_KDTREE=0
    indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    searchParams=dict(checks=50)
    flann=cv2.FlannBasedMatcher(indexParams,searchParams)
    sampleImage=cv2.imread(path1,0)
    queryImage=cv2.imread(path2,0)
    kp1, des1 = sift.detectAndCompute(sampleImage, None) #提取样本图片的特征
    kp2, des2 = sift.detectAndCompute(queryImage, None) #提取比对图片的特征
    matches=flann.knnMatch(des1,des2,k=2) #匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
    (matchNum,matchesMask)=getMatchNum(matches,0.9) #通过比率条件，计算出匹配程度
    matchRatio=matchNum/len(matches)
    return(matchRatio)

def compare(img1,img2):
    lpips = lpipsmetric(img1, img2)
    dists = distsmetric(img1, img2)
    ssim = ssimmetric(img1, img2)
    sift = siftmetric(img1, img2)
    #print("%.5f"%(lpips),"%.5f"%(dists),"%.5f"%(ssim),"%.5f"%(sift))
    return("%.5f"%(lpips),"%.5f"%(dists),"%.5f"%(ssim),"%.5f"%(sift))

for m in range(1,101):
    path1 = ("./imgs/testimg1/%i.png"%(m))
    img1 = cv2.imread(path1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    path2 = ("./imgs/testimg2/%i.png"%(m))
    img2 = cv2.imread(path2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    with open("slices.csv","a",newline="") as datacsv:
        #dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
        csvwriter = csv.writer(datacsv,dialect = ("excel"))
        #csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
        csvwriter.writerow(compare(img1,img2)) #["%.5f"%(lpips),"%.5f"%(dists),"%.5f"%(msssim),"%.5f"%(sift)]
        #csvwriter.writerow("\n")
    
