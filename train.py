# Copyright 2018, Satoshi Ikehata, National Institute of Informatics (sikehata@nii.ac.jp)
w = 32 # size of observation map

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
from mymodule import cnn_models as cm

isParallel = True # one if multi gpu is available

if K.image_data_format() == 'channels_first': # theano
    model = cm.get_densenet_2d_channel_first_2dense(w,w)
else: # tensorflow or cntk
    model = cm.get_densenet_2d_channel_last_2dense(w,w)
if isParallel:
    parallel_model = multi_gpu_model(model, gpus=3)
    parallel_model.compile(optimizer=keras.optimizers.Adam(), loss='mean_squared_error')
else:
    model.compile(optimizer=keras.optimizers.Adam(), loss='mean_squared_error')

datagenerated = '/media/sikehata/e4608bc5-8fbc-4efe-a6f3-7ac7f270d0ea/BlenderRendering/DataGenerated' # path to the training dataset
objlist = sorted(os.listdir(datagenerated + '/PRPS'))
epochs = 1
min_err = 1000
rotdivin = 10
rotdivon = 10
datasplit = 3 # 25/x should be integer

subsetsize = np.int32(len(objlist)/datasplit)
for k in range(10):
    datalist = []
    for p in range(datasplit):
        print('%d-th loop' % (k+1), '%d' % (p+1) + '/' + '%d' % datasplit)
        SList = []
        NList = []
        for q in range(subsetsize):

            objroot = datagenerated + '/PRPS_Diffuse'
            dirname = 'images_diffuse'
            datapath = [objroot + '/' + '%s' % objlist[subsetsize*p+q]]
            print(datapath)
            S,M,N = dio.prep_data_2d_from_images_cycles(datapath, dirname, 1, w, rotdivin, rotdivon)
            SList.append(S.copy())
            NList.append(N.copy())
            del S, M, N

            objroot = datagenerated + '/PRPS'
            dirname = 'images_specular'
            datapath = [objroot + '/' + '%s' % objlist[subsetsize*p+q]]
            print(datapath)
            S,M,N = dio.prep_data_2d_from_images_cycles(datapath, dirname, 0.5, w, rotdivin, rotdivon)
            SList.append(S.copy())
            NList.append(N.copy())
            del S, M, N

            objroot = datagenerated + '/PRPS'
            datapath = [objroot + '/' + '%s' % objlist[subsetsize*p+q]]
            print(datapath)
            dirname = 'images_metallic'
            S,M,N = dio.prep_data_2d_from_images_cycles(datapath, dirname, 0.5, w, rotdivin, rotdivon)
            SList.append(S.copy())
            NList.append(N.copy())
            del S, M, N

        SList = np.float32(np.concatenate(SList, axis=0))
        NList = np.float32(np.concatenate(NList, axis=0))

        if  isParallel:
            hist = parallel_model.fit(SList, NList, batch_size= 768, epochs= epochs, verbose=1, shuffle=True, validation_split=0.1)
        else:
            hist = model.fit(SList, NList, batch_size= 1024, epochs= epochs, verbose=1, shuffle=True, validation_split=0.1)

            model.save('weight_and_model_user.hdf5')
            print('Model Updated!!')
        del SList,NList
