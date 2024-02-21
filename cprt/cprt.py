import cv2
import imageio
import math
import os
import torch

import elpips
import lpips
import pytorch_ssim
#from pytorch_msssim import ms_ssim
from DISTS_pytorch import DISTS

from IPython import embed
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


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

'''
def msssimmetric(img1, img2):
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)
    image1 = torch.from_numpy(np.rollaxis(image1, 2)).float().unsqueeze(0)/255.0
    image2 = torch.from_numpy(np.rollaxis(image2, 2)).float().unsqueeze(0)/255.0
    image1 = (image1 + 1) / 2  # [-1, 1] => [0, 1]
    image2 = (image2 + 1) / 2
    ms_ssim_val = ms_ssim( image1, image2, data_range=1, size_average=False )
    return ms_ssim_val
'''

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

def compare_images(img1, img2):
    # index for the images
    lpips = lpipsmetric(img1, img2)
    dists = distsmetric(img1, img2)
    ssim = ssimmetric(img1, img2)
    sift = siftmetric(img1, img2)
    sim = 0.358479*lpips + 0.251745*dists + 0.247436*ssim + 0.14234*sift
    return sim

def show_images(img1, img2, title):
    # index for the images
    lpips = lpipsmetric(img1, img2)
    dists = distsmetric(img1, img2)
    ssim = ssimmetric(img1, img2)
    sift = siftmetric(img1, img2)
    sim = 0.358479*lpips + 0.251745*dists + 0.247436*ssim + 0.14234*sift
    # setup the figure
    fig = plt.figure(title)
    #plt.suptitle("LPIPS: %.2f, DISTS: %.2f, SSIM: %.2f, SIFT: %.2f, similarity: %.2f" % (lpips, dists, ssim, sift, sim))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img1)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img2, cmap=plt.cm.gray)
    plt.axis("off")
    # show the images
    plt.show()
    
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize[1]):
        for x in range(0, image.shape[1], stepSize[0]):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])



if __name__ == '__main__':
    image1 = cv2.imread('./imgs/1/14.png')
    image2 = cv2.imread('./imgs/2/14.png')
    # 自定义滑动窗口的大小
    w1 = image1.shape[1]
    h1 = image1.shape[0]
    w2 = image2.shape[1]
    h2 = image2.shape[0]
    #winW, winH和stepSize可自行更改
    #(winW, winH) = (int(3*w/4),int(2*w/3))
    (winW1, winH1) = (200,200)
    stepSize = (20, 20)
    cnt1 = 0
    (winW2, winH2) = (200,200)
    cnt2 = 0
    for (x1, y1, window1) in sliding_window(image1, stepSize=stepSize, windowSize=(winW1, winH1)):
        # if the window does not meet our desired window size, ignore it
        if window1.shape[0] != winH1 or window1.shape[1] != winW1:
            continue
        # since we do not have a classifier, we'll just draw the window
        clone1 = image1.copy()
        cv2.rectangle(clone1, (x1, y1), (x1 + winW1, y1 + winH1), (0, 255, 0), 2)
        clone1 = cv2.cvtColor(clone1, cv2.COLOR_BGR2RGB)
        slice1 = image1[y1:y1+winH1,x1:x1+winW1]
        path1 = ("./slices/slices01./%i%i.png"%(x1,y1))
        img1 = cv2.imread(path1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        for (x2, y2, window2) in sliding_window(image2, stepSize=stepSize, windowSize=(winW2, winH2)):
            # if the window does not meet our desired window size, ignore it
            if window2.shape[0] != winH2 or window2.shape[1] != winW2:
                continue
            # since we do not have a classifier, we'll just draw the window
            clone2 = image2.copy()
            cv2.rectangle(clone2, (x2, y2), (x2 + winW2, y2 + winH2), (0, 255, 0), 2)
            clone2 = cv2.cvtColor(clone2, cv2.COLOR_BGR2RGB)
            slice2 = image2[y2:y2+winH2,x2:x2+winW2]
            path2 = ("./slices/slices02./%i%i.png"%(x2,y2))
            img2 = cv2.imread(path2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            simi = compare_images(img1,img2)
            print(simi)
            if  simi>0.384:
                show_images(clone1, clone2, "similarity images")
            else:
                continue
            cnt2 = cnt2 + 1
        cnt1 = cnt1 + 1

