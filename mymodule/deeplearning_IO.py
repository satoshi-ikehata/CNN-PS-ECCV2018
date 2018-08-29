import h5py
import numpy as np
from scipy.io import loadmat
from operator import itemgetter
import math
import scipy as sp
import cv2
import matplotlib.pyplot as plt
import os, sys
import time
import multiprocessing


import random

# Generate Observation Map
def func(theta, m, I, imax, L, w, N, anglemask):
    print('*',end='')
    rotmat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    p = 0.5*(L[:,0]+1)*(w-1) #x 0:w-1
    q = 0.5*(L[:,1]+1)*(w-1) #y 0:w-1
    x = [p-0.5*(w-1), q-0.5*(w-1)]
    x_ = np.dot(rotmat, x)
    p = x_[0,:]+0.5*(w-1);
    q = x_[1,:]+0.5*(w-1);
    p = np.int32(p)
    q = np.int32(q)
    light_idx = q*w + p # 0:w*w-1
    x = [N[:,0], N[:,1]]
    x_ = np.dot(rotmat, x)
    pn = x_[0,:];
    qn = x_[1,:];
    normal = [np.transpose(pn), np.transpose(qn), N[:,2]]
    normal = np.transpose(normal)
    temp = I*anglemask/np.transpose(imax)
    embed = np.zeros((m, w*w), np.float32)
    embed[:, light_idx] = temp
    embed = np.reshape(embed, (m, w, w))
    mask = np.zeros((m, w*w), np.bool_)
    mask[:, light_idx] = anglemask
    mask = np.reshape(mask, (m, w, w))
    return embed, mask, normal, rotmat

def wrapper(args):
    return func(*args)


# for multi core cpu
def light_embedding_2d_rot_invariant_multi(I, imax, L, w, N, div, isRandomThresh):

    m = I.shape[0]
    rows = w
    cols = w
    embed_rot = []
    normal_rot = []
    mask_rot = []
    rot = []

    anglemask = np.zeros((I.shape[0],I.shape[1]),np.float32)
    for k in range(I.shape[0]): # numpixel
        angle1 = 180*np.arccos(L[:,2])/np.pi
        if isRandomThresh == True:
            tgt = np.where(angle1<random.randint(20,90))
            tgtrandom = np.random.permutation(tgt[0])
            tgt = tgtrandom[:random.randint(50,np.min([1000,L.shape[0]]))]
        else:
            tgt = np.where(angle1<90)
        anglemask[k,tgt] = 1

    n = multiprocessing.cpu_count()
    p = multiprocessing.Pool(n)

    params = [(np.pi*(i*360.0/div)/180, m, I, imax, L, w, N, anglemask) for i in range(np.int32(div))]
    result = p.map(wrapper, params)
    p.close()

    embed_list = []
    mask_list = []
    nml_list = []
    rot_list = []
    for i in range(div):
        embed_list.append(result[i][0].copy())
        mask_list.append(result[i][1].copy())
        nml_list.append(result[i][2].copy())
        rot_list.append(result[i][3].copy())

    embed_list = np.array(embed_list)
    embed_list = np.transpose(embed_list, (1,0,2,3))

    mask_list = np.array(mask_list)
    mask_list = np.transpose(mask_list, (1,0,2,3))

    nml_list = np.array(nml_list)
    nml_list = np.transpose(nml_list, (1,0,2))

    del result,anglemask

    return np.array(embed_list), np.array(mask_list), np.array(nml_list), np.array(rot_list), rows, cols

# for single core cpu
def light_embedding_2d_rot_invariant(I, imax, L, w, N, div, isRandomThresh):

    m = I.shape[0]

    embed_rot = []
    normal_rot = []
    mask_rot = []
    rot = []
    count = 0

    anglemask = np.zeros((I.shape[0],I.shape[1]),np.float32)
    for k in range(I.shape[0]):

        angle1 = 180*np.arccos(L[:,2])/np.pi
        if isRandomThresh == True:
            tgt = np.where(angle1<random.randint(20,90))
            tgtrandom = np.random.permutation(tgt[0])
            tgt = tgtrandom[:random.randint(50,np.min([1000,L.shape[0]]))]
        else:
            tgt = np.where(angle1<90)
        anglemask[k,tgt] = 1

    for k in range(div):
        theta = k*360/div
        if theta < 360:
            count = count + 1
            theta = np.pi*theta/180
            rotmat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            p = 0.5*(L[:,0]+1)*(w-1) #x 0:w-1
            q = 0.5*(L[:,1]+1)*(w-1) #y 0:w-1
            x = [p-0.5*(w-1), q-0.5*(w-1)]
            x_ = np.dot(rotmat, x)
            p = x_[0,:]+0.5*(w-1);
            q = x_[1,:]+0.5*(w-1);
            p = np.int32(p)
            q = np.int32(q)
            light_idx = q*w + p # 0:w*w-1

            x = [N[:,0], N[:,1]]
            x_ = np.dot(rotmat, x)
            pn = x_[0,:];
            qn = x_[1,:];
            normal = [np.transpose(pn), np.transpose(qn), N[:,2]]
            normal = np.transpose(normal)

            temp = I*anglemask/np.transpose(imax)
            embed = np.zeros((m, w*w), np.float32)
            embed[:, light_idx] = temp
            embed = np.reshape(embed, (m, w, w))
            embed_rot.append(embed.copy())
            mask = np.zeros((m, w*w), np.bool_)
            mask[:, light_idx] = anglemask
            mask = np.reshape(mask, (m, w, w))
            mask_rot.append(mask.copy())

            normal_rot.append(normal.copy())
            rot.append(rotmat.copy())

            del embed, temp, normal, mask

    rows = w
    cols = w

    embed_rot = np.array(embed_rot)
    embed_rot = np.transpose(embed_rot, (1,0,2,3))

    mask_rot = np.array(mask_rot)
    mask_rot = np.transpose(mask_rot, (1,0,2,3))

    normal_rot = np.array(normal_rot)
    normal_rot = np.transpose(normal_rot, (1,0,2))

    return np.array(embed_rot), np.array(mask_rot), np.array(normal_rot), np.array(rot), rows, cols


# main function for generating the observation map
def light_embedding_main(Iv, Nv, L, w, rotdiv, validind, isRandomThresh):

    imax = np.amax(Iv,axis=1) # for entire image
    valid = np.intersect1d(validind, np.where(imax>0))
    Iv = Iv[valid,:]
    Nv = Nv[valid,:]
    imax = imax[valid]
    if rotdiv > 1:
        embed, mask, nm, rot, rows, cols = light_embedding_2d_rot_invariant_multi(Iv, [imax], L, w, Nv, rotdiv, isRandomThresh)
    else:
        embed, mask, nm, rot, rows, cols = light_embedding_2d_rot_invariant(Iv, [imax], L, w, Nv, rotdiv, isRandomThresh)

    embed = np.reshape(embed, (embed.shape[0]*embed.shape[1],w,w))
    embed = np.reshape(embed, (embed.shape[0],1,w,w))
    mask = np.reshape(mask, (mask.shape[0]*mask.shape[1],w,w))
    mask = np.reshape(mask, (mask.shape[0],1,w,w))
    nm = np.reshape(nm, (nm.shape[0]*nm.shape[1],3))
    return embed, mask, nm

# prepare observation map for cyclesPS dataset (for training)
def prep_data_2d_from_images_cycles(dirlist, dirname, scale, w, rotdiv_in, rotdiv_on):
    S = []
    M = []
    N = []
    for d in dirlist:
        dirpath = d

        images_dir = dirpath + '/' + dirname
        normal_path = dirpath + '/' + 'gt_normal.tif'
        inboundary_path = dirpath + '/' + 'inboundary.png'
        onboundary_path = dirpath + '/' + 'onboundary.png'

        # read ground truth surface normal
        nml = np.float32(cv2.imread(normal_path,-1))/65535.0 # [-1,1]
        nml = nml[:,:,::-1]
        nml = 2*nml-1
        nml = cv2.resize(nml, None, fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
        nShape = np.shape(nml)
        height = nShape[0]
        width = nShape[1]

        # read mask images_metallic
        inboundary = cv2.imread(inboundary_path,-1)
        inboundary = cv2.resize(inboundary, None, fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
        inboundary = np.where(inboundary>0)
        inboundary_ind = inboundary[0]*height + inboundary[1]
        onboundary = cv2.imread(onboundary_path,-1)
        onboundary = cv2.resize(onboundary, None, fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
        onboundary = np.where(onboundary>0)
        onboundary_ind = onboundary[0]*height + onboundary[1]

        # read light filenames
        f = open(dirpath + '/' 'light.txt')
        data = f.read()
        f.close
        lines = data.split('\n')
        numLight = len(lines)-1 # the last line is empty (how to fix it?)

        L = np.zeros((numLight,3), np.float32)
        for i,l in enumerate(lines):
            s = l.split(' ')
            if len(s) == 3:
                L[i,0] = float(s[0])
                L[i,1] = float(s[1])
                L[i,2] = float(s[2])

        # read images
        I = np.zeros((numLight, height, width), np.float32)

        for i in range(numLight):
            if i % np.floor(numLight/10) == 0:
                print('.',end='')

            image_path = images_dir + '/' + '%05d.tif' % i

            cv2_im = cv2.imread(image_path, -1)/65535.0
            cv2_im = (cv2_im[:,:,0] + cv2_im[:,:,1] + cv2_im[:,:,2])/3
            cv2_im = cv2.resize(cv2_im, (height,width), interpolation = cv2.INTER_NEAREST)
            I[i,:,:] = cv2_im


        Iv = np.reshape(I,(numLight, height*width))
        Iv = np.transpose(Iv)

        Nv = np.reshape(nml,(height*width,3))

        embed_in, mask_in, nm_in = light_embedding_main(Iv, Nv, L, w, rotdiv_in, inboundary_ind, True)
        embed_on, mask_on, nm_on = light_embedding_main(Iv, Nv, L, w, rotdiv_on, onboundary_ind, True)

        embed = []
        embed.append(embed_in.copy())
        embed.append(embed_on.copy())
        embed = np.concatenate(embed, axis=0 )

        mask = []
        mask.append(mask_in.copy())
        mask.append(mask_on.copy())
        mask = np.concatenate(mask, axis=0 )

        nm = []
        nm.append(nm_in.copy())
        nm.append(nm_on.copy())
        nm = np.concatenate(nm, axis=0 )


        S.append(embed.copy())
        M.append(mask.copy())
        N.append(nm.copy())
        print('')

        del embed_in, mask_in, nm_in
        del embed_on, mask_on, nm_on
        del embed, mask, nm, I, Iv, Nv

    S = np.concatenate(S, axis=0 )
    M = np.concatenate(M, axis=0 )
    N = np.concatenate(N, axis=0 )

    S = np.reshape(S, (S.shape[0], S.shape[2], S.shape[3], 1))
    M = np.reshape(M, (M.shape[0], M.shape[2], M.shape[3], 1))
    return np.array(S), np.array(M), np.array(N)

# prepare observation maps for test data (i.e., DiLiGenT dataset)
def prep_data_2d_from_images_test(dirlist, scale, w, rotdiv, index=-1):

    SList = []
    NList = []
    RList = []
    IDList = []
    SizeList = []
    for d in dirlist:
        print('load' + '%s' % d)
        S = []
        N = []
        dirpath = d
        images_dir = dirpath
        normal_path = dirpath + '/' + 'normal.txt'
        mask_path = dirpath + '/' + 'mask.png'

        # get image imgSize
        image_path = images_dir + '/' + '001.png'
        cv2_im = cv2.imread(image_path, -1)
        nShape = np.shape(cv2_im)
        height = nShape[0]
        width = nShape[1]

        # read ground truth surface normal
        f = open(normal_path)
        data = f.read()
        f.close
        lines = np.float32(np.array(data.split('\n')))
        nml = np.reshape(lines, (height,width,3))
        nml = cv2.resize(nml, None, fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
        # nml = np.flipud(nml) # when test on Harvest, the surface noraml needs to be fliped upside down
        nShape = np.shape(nml)
        height = nShape[0]
        width = nShape[1]

        # uncomment if you want to see the ground truth normal map
        # plt.figure(figsize=(16,16))
        # plt.imshow(np.uint8(127*(nml+1)))
        # plt.axis('off')
        # plt.show()

        # read mask
        mask = cv2.imread(mask_path,-1)
        mask = cv2.resize(mask, None, fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
        validsub = np.where(mask>0)
        validind = validsub[0]*width + validsub[1]

        # read light directions
        f = open(dirpath + '/' 'light_directions.txt')
        data = f.read()
        f.close
        lines = data.split('\n')
        numLight = len(lines)-1 # the last line is empty (how to fix it?)

        L = np.zeros((numLight,3), np.float32)
        for i,l in enumerate(lines):
            s = l.split(' ')
            if len(s) == 3:
                L[i,0] = float(s[0])
                L[i,1] = float(s[1])
                L[i,2] = float(s[2])

        # read light intensities
        f = open(dirpath + '/' 'light_intensities.txt')
        data = f.read()
        f.close
        lines = data.split('\n')

        Li = np.zeros((numLight,3), np.float32)
        for i,l in enumerate(lines):
            s = l.split(' ')
            if len(s) == 3:
                Li[i,0] = float(s[0])
                Li[i,1] = float(s[1])
                Li[i,2] = float(s[2])


        if index == -1:
            index = range(0, numLight)

        L = L[index,:]
        Li = Li[index,:]
        numLight = len(index)

        # read images
        I = np.zeros((numLight, height, width), np.float32)

        for i, idx in enumerate(index):
            if i % np.floor(numLight/10) == 0:
                print('.',end='')
            image_path = images_dir + '/' + '%03d.png' % (idx + 1)
            cv2_im = cv2.imread(image_path, -1)/65535.0
            cv2_im = (cv2_im[:,:,0]/Li[i,0] + cv2_im[:,:,1]/Li[i,1] + cv2_im[:,:,2]/Li[i,2])/3
            cv2_im = cv2.resize(cv2_im, None, fx = scale, fy = scale,interpolation = cv2.INTER_NEAREST)
            I[i,:,:] = cv2_im

        Iv = np.reshape(I,(numLight, height*width))
        Iv = np.transpose(Iv)
        Nv = np.reshape(nml,(height*width,3))

        imax = np.amax(Iv,axis=1) # for entire image
        valid = np.intersect1d(validind, np.where(imax>0))
        Iv = Iv[valid,:]
        Nv = Nv[valid,:]
        imax = imax[valid]
        embed_list = []
        embed, mask, nm, rot, rows, cols = light_embedding_2d_rot_invariant(Iv, [imax], L, w, Nv, rotdiv, False)
        SList.append(embed)
        RList.append(rot)
        NList.append(nm)
        IDList.append(valid)
        SizeList.append((height,width))

        print('')
    return np.array(SList), np.array(NList), np.array(RList), np.array(IDList), np.array(SizeList)


# Test and evaluate network
def TestNetwork(model, Sv,Nv,Rv,IDv,Szv,showFig, isTensorFlow):
    numData = len(Sv)
    for i in range(numData):
        S = Sv[i]
        N = Nv[i]
        R = Rv[i]
        ID = IDv[i]
        height = Szv[i,0]
        width  = Szv[i,1]
        rotdiv = S.shape[1]
        NestList = []
        for r in range(rotdiv):
            embed_div = S[:,r,:,:]
            if isTensorFlow:
                embed_div = np.reshape(embed_div, (embed_div.shape[0], embed_div.shape[1], embed_div.shape[2], 1))
            else:
                embed_div = np.reshape(embed_div, (embed_div.shape[0], 1, embed_div.shape[1], embed_div.shape[2]))
            # predict
            outputs=model.predict(embed_div)
            Nest = np.zeros((height*width,3), np.float32)
            error = 0
            Err = np.zeros((height*width,3), np.float32)
            rot = R[r,:,:]
            # N = np.zeros()
            for k in range(len(ID)):
                # n = outputs[k,:];
                n = np.zeros((2,1),np.float32)
                n[0] = outputs[k,0]
                n[1] = outputs[k,1]
                n = np.dot(np.linalg.inv(rot),n)
                n = [n[0,0],n[1,0],outputs[k,2]]
                n = n/np.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])
                nt = N[k,0,:];
                Nest[ID[k],:] = n
                for l in range(3):
                    Err[ID[k],l] = 180*math.acos(min(1,abs(n.dot(np.transpose(nt)))))/math.pi
                error = error + 180*math.acos(min(1,abs(n.dot(np.transpose(nt)))))/math.pi
            print('%d ' % i + '[Angle %d] Ave.Error = %.2f ' % (r,(error/len(ID))))
            NestList.append(Nest.copy())

        NestMean = np.mean(NestList,axis=0)
        Nest = np.zeros((height*width,3), np.float32)
        error = 0
        Err = np.zeros((height*width,3), np.float32)
        for k in range(len(ID)):
            # n = outputs[k,:];
            n = NestMean[ID[k],:]
            n = n/np.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])
            nt = N[k,0,:];
            Nest[ID[k],:] = n
            for l in range(3):
                Err[ID[k],l] = 180*math.acos(min(1,abs(n.dot(np.transpose(nt)))))/math.pi
            error = error + 180*math.acos(min(1,abs(n.dot(np.transpose(nt)))))/math.pi

        if rotdiv >= 2:
            print('%s ' % i + '[Mean] Ave.Error = %.2f ' % (error/len(ID)))

        Err = np.reshape(Err,(height,width,3))
        Nest = np.reshape(Nest, (height,width,3))

        if showFig == True:
            plt.figure(figsize=(16,16))
            plt.imshow(np.concatenate((np.uint8(127*(Nest+1)),5*np.uint8(Err)), axis=1))
            plt.axis('off')
            plt.show()
