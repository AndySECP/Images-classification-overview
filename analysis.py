#analysis
#author: Andy
#date: 05/10/2019

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import skimage
from skimage import data
from skimage import io
import os, os.path
import seaborn as sns
import cv2
from scipy import linalg
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def read_organize_data(file_path):

    #Fill this function out, should return a dataframe with picture object, and correct encoding
    objects = ['Airplanes', 'Bear', 'Blimp', 'Comet', 'Crab', 'Dog', 'Dolphin', 'Giraffe',\
               'Goat', 'Gorilla', 'Kangaroo', 'Killer-Whale', 'Leopards', 'Llama', 'Penguin',\
               'Porcupine', 'Teddy-Bear', 'Triceratops', 'Unicorn', 'Zebra']
    pictures = []    
    encoding = []
    directory_path = file_path
    for element in objects:        
        new_dir = directory_path+'\\'+element.lower()
        for filename in os.listdir(new_dir):
            pictures.append(io.imread(new_dir+'\\'+filename))
            encoding.append(element) 
    dat = pd.DataFrame({'Pictures':pictures, 'Encoding':encoding})

    return dat    


def extract_features(image, vector_size=32):   

    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])

    except cv2.error as e:
        print ('Error: ')
        return None

    return dsc


def get_pca_tsne(data):

    pca = PCA(n_components=2).fit_transform(data)
    tse = TSNE(n_components=2).fit_transform(data)
    plt.figure(figsize=(18,6))
    plt.subplot(1,2,1)
    plt.title('PCA Visualization', size=23)
    plt.scatter(pca[:,0], pca[:,1])
    #plt.ylabel('Color | {}'.format(metric), size=20)
    plt.xticks([])
    plt.yticks([])
    #plt.colorbar()
    plt.subplot(1,2,2)
    plt.title('TSNE Visualization', size=23)
    plt.scatter(tse[:,0], tse[:,1])
    #plt.ylabel('Color | {}'.format(metric), size=20)
    plt.xticks([])
    plt.yticks([])
    #plt.colorbar()
    plt.tight_layout()
    plt.show()

    return



def get_pca_tsne2(data, metric):

    pca = PCA(n_components=2).fit_transform(data)
    tse = TSNE(n_components=2).fit_transform(data)
    plt.figure(figsize=(18,6))
    plt.subplot(1,2,1)
    plt.title('PCA Visualization', size=23)
    plt.scatter(pca[:,0], pca[:,1], c=data[metric])
    plt.ylabel('Color | {}'.format(metric), size=20)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.title('TSNE Visualization', size=23)
    plt.scatter(tse[:,0], tse[:,1], c=data[metric])
    plt.ylabel('Color | {}'.format(metric), size=20)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    return


def svd_rgb(image):

    single_color_mat=image[:,:,0]
    _, s, _ = linalg.svd(single_color_mat)
    mean_sing=np.mean(s)

    return pd.Series(mean_sing)


def svd_rgb2(image):

    single_color_mat2=image[:,:,1]
    _, s2, _ = linalg.svd(single_color_mat2)
    mean_sing2=np.mean(s2)

    return pd.Series(mean_sing2)


def svd_rgb3(image):

    single_color_mat3=image[:,:,2]
    _, s3, _ = linalg.svd(single_color_mat3)
    mean_sing3=np.mean(s3)

    return pd.Series(mean_sing3)


def count_blue(image):

    single_color_mat=image[:,:,2]
    h,w = single_color_mat.shape
    blue_prop= (single_color_mat > 170).sum()/(h*w)

    return pd.Series(blue_prop)