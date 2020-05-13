import numpy as np
import rasterio
import glob, os
import os.path  
import skimage
import skimage.morphology
import skimage.measure
import pandas as pd
import geopandas
from shapely.geometry import Polygon
from itertools import compress


def load_bands(path,bands,verbose=0):
    """
    load_bands(path,bands):
    
    This function loads images of type .jp2 from a IMG_DATA folder definded from "path".
    The images are loaded as numpy arrays(NxM) in a list of size "bands". 
     
    Supported for L2A products
    object cant be returned for L1C products
    
    Version 1.2
    ================================================
    Input:
        path(str) = path img IMG_DATA folder.
        bands(list of str) = A list of strings containing the band names.
    
    Output: 
        img_container(list) = A list of size (bands) containing each band as a numpy array image of same size.
        rasterio_obj(list) = A list of rasterio objects. Usefull for coordinate transformation. 
                            !!! Only works frot R10m rest is returned as None "                          
    
    Example:
        get list of images:
        images = load_bands('F:\\S2_billeder\\..\\IMG_DATA',bands=['B03','B04','B05','B01'])
    
    Author:
        Krisitan Sørensen
        Revised: May 2020
    
    """
    ####################################
    # Assertions
    ####################################    
    assert len(bands)>0,"No bands are given"
    
    Driver='JP2OpenJPEG'
    
    #count number og .jp2 files in .IMG folder. If there is some, if old format
    # if none, then theres subfolders and new format.
    folder = glob.glob(os.path.join(path,"*.jp2"))
    if (len(folder)>0):
        data_type = 'L1C'
        print('(Load_bands message) Image is L1C format.')
    else:
        data_type = 'L2A'
        print('(Load_bands message) Image is L2A format.')
        assert os.path.exists(os.path.join(path,'R10m')),"band folders doesn't exist in path folder"
        
    
    
    ####################################
    #Loading the bands
    ####################################
    
    img_container = []
    rasterio_obj = []
    #in data_type='new', the images are found in different subfolders.
    if (data_type =='L2A'):
        for i in range(len(bands)):
            #For bands B02,B03,B04 and B08, chaning path to R10m(10 m resolutuon)
            if (bands[i]=='B02') or (bands[i]=='B03') or (bands[i]=='B04')  or (bands[i]=='B08') or (bands[i]=='AOT') or (bands[i]=='TCI') or (bands[i]=='WVP'):    
                img_path = os.path.join(path,'R10m')
                img_path = os.path.join(img_path,'')
                img_extend = bands[i]+"_10m.jp2"
                for file in os.listdir(img_path):
                    if file.endswith(img_extend):
                        img = rasterio.open(img_path+file, driver=Driver)
                        rasterio_obj.append(img)
                        img_container.append((np.asarray(img.read(window=((0,img.shape[0]),(0,img.shape[1])))))[0,:,:])
                        if verbose==1:
                            print(bands[i],' loaded')
                                   
            if (bands[i]=='B05') or (bands[i]=='B06') or (bands[i]=='B07')  or (bands[i]=='B11') or (bands[i]=='B12') or (bands[i]=='SCL'):
                img_path = os.path.join(path,'R20m')
                img_path = os.path.join(img_path,'')
                img_extend = bands[i]+"_20m.jp2"
                for file in os.listdir(img_path):
                    if file.endswith(img_extend):
                        img = rasterio.open(img_path+file, driver=Driver)
                        rasterio_obj.append(img)
                        img = (np.asarray(img.read(window=((0,img.shape[0]),(0,img.shape[1])))))[0,:,:]
                        img = np.repeat(img, 2, axis=0)
                        img = np.repeat(img, 2, axis=1)
                        img = img.astype('float')
                        img_container.append(img)
                        if verbose==1:
                            print(bands[i],' loaded')
                        
            # 
            if (bands[i]=='B01') or (bands[i]=='B09'):
                # Aerosol Optical Thickness AOT. 
                img_path = os.path.join(path,'R60m')
                img_path = os.path.join(img_path,'')
                img_extend = bands[i]+"_60m.jp2"
                for file in os.listdir(img_path):
                    if file.endswith(img_extend):
                        img = rasterio.open(img_path+file, driver=Driver)
                        rasterio_obj.append(img)
                        img = (np.asarray(img.read(window=((0,img.shape[0]),(0,img.shape[1])))))[0,:,:]
                        img = np.repeat(img, 6, axis=0)
                        img = np.repeat(img, 6, axis=1)
                        img = img.astype('float')
                        img_container.append(img)
                        if verbose==1:
                            print(bands[i],' loaded')
    
    #if data_type ='old', all .jp2 files is in .IMG folder.
    elif (data_type=='old'):
        for i in range(len(folder)):
            for j in range(len(bands)):
                if folder[i].endswith(bands[j]+'.jp2'):
                    img = rasterio.open(folder[i], driver=Driver)
                    rasterio_obj.append(None)
                    img = (np.asarray(img.read(window=((0,img.shape[0]),(0,img.shape[1])))))[0,:,:]
                    
                    if (bands[j]=='B05') or (bands[j]=='B06') or (bands[j]=='B07')  or (bands[j]=='B11') or (bands[j]=='B12') or (bands[j]=='SCL'):
                        img = np.repeat(img, 2, axis=1)
                        img = np.repeat(img, 2, axis=0)
                        img_container.append(img)
                        if verbose==1:
                            print(bands[j],' loaded')    
                    elif (bands[j]=='B01') or (bands[j]=='B09') or (bands[j]=='B10'):
                        img = np.repeat(img, 6, axis=1)
                        img = np.repeat(img, 6, axis=0)
                        img_container.append(img)
                        if verbose==1:
                            print(bands[j],' loaded')
                     
        
        
    return img_container, rasterio_obj








def rgb_img(r,g,b):
    """
    rgb color representation.
    just illustration.. 
    (NxMx3) -> (NxMx1)
    """
    
    #color_image = np.stack((r*100,(g*42), (b*10)), axis=2)
    color_image = np.stack((r,g, b), axis=2)
    color_image = color_image/np.quantile(color_image,0.99)
    
    return color_image


def img_clip(img,cliphigh=0.9,cliplow=0.1):
    """
    Just at simple clipping used for our stuff.
    Maybe use CFAR later? (not)
    
    """
    for i in range(len(img)):
        img[i] = np.clip(img[i], np.quantile(img[i],cliplow), np.quantile(img[i],cliphigh))
        
    img_clipped = img
    
    return img_clipped

import numpy as np
import cv2

