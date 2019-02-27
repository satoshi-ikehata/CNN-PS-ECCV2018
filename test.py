# Copyright 2018, Satoshi Ikehata, National Institute of Informatics (sikehata@nii.ac.jp)

import importlib
import numpy as np
import pydot
import os
import keras
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.utils import multi_gpu_model
from mymodule import deeplearning_IO as dio
from PIL import Image
import matplotlib.pyplot as plt

# NOTICE!!
# bearPNG has problem in first 20 images, so please discard first 20 information from Sv Nv Rv IDv Szv if you want the result from the paper
# harvestPNG has problem in the order of values in normal.txt, so you need flip the surface normal map upside down

def main():
    isParallel = True # one if multi gpu is available
    dirlist = []
    diligent = '/media/sikehata/e4608bc5-8fbc-4efe-a6f3-7ac7f270d0ea/BlenderRendering/Data/DiLiGenT/pmsData'

    # Set the directory list
    # objlist = ('ballPNG','bearPNG','buddhaPNG','catPNG','cowPNG','gobletPNG','harvestPNG','pot1PNG','pot2PNG','readingPNG')
    objlist = (['bearPNG'])



    for f in objlist:
        dirlist.append(diligent + '/' + f)
    print(dirlist)

    # Prepare images
    scale = 1 # downsize the image by this scale
    w = 32 # size of observation map
    K = 10 # the number of different rotations for the rotational pseudo-invariance


    [Sv, Nv, Rv, IDv, Szv] = dio.prep_data_2d_from_images_test(dirlist, scale, w, K) # Comment this line when runnning test on bearPNG
    # [Sv, Nv, Rv, IDv, Szv] = dio.prep_data_2d_from_images_test(dirlist, scale, w, K, index = range(20, 96)) # Uncomment this if you want to use the subset of images (in this case, 20-th to 96-th images are input)
    # Load pretrained model
    model = load_model('weight_and_model.hdf5')

    # Test network
    dio.TestNetwork(model,Sv,Nv,Rv,IDv,Szv,showFig=1,isTensorFlow=1)

if __name__ == '__main__':
    main()
