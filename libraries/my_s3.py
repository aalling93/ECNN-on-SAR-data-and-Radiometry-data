import os
import sys
import math
import numpy as np
import gdal
import rasterio
import matplotlib.pyplot as plt
from osgeo import gdal


#from matplotlib import pyplot as plt
#from osgeo import osr
#import ogr
#import osr
#import gdalnumeric
#import gdalconst
#import pandas as pd
#import rasterio as rio
#import geopandas as gpd
#import geopandas # the GEOS-based vector package
#import contextily # the package for fetching basemaps
#from glob import glob
#from rasterio.plot import show
#from rasterio import plot as rioplot
#sys.path.append('C:\\Users\\krist\\Documents\\Syntese\\code')
#import my_tools as tools


def get_img(path,data_type='ENVI',verbose=0):
    """
    s3.get_img():
    
    This fucntion loads the Sentinel-3 images into python using the gdal library.
    The function takes all image bands in the path, and saves the raster information. the image as an array, and all the other information.
    
    Input:
        path[str]: String of path of tif file.
        data_type[str]: which driver shoud be used. Default and suggested is ENVi.
        verbose[int]: amount of information that will be printed. 0=no info, 1=info
        
    Output:
        images[list]: list of arrays. 
        raster[obj]: object with raster information
        extend[list]: list of geoprahic information. For e.g. plotting.
        names[list]: list of string names.
        extra[list]: list of advanced info. Here, offset, rotation and more is given.
        
    Example:
        images,raster,extend, names, extra = s1.get_img('home/syntese/data/s3/s3_23.tif',data_type='ENVI')
        
    Author:
        Kristian Aalling Soerensen
        May 2020
        kristian.sorensen@hotmail.com
    """
    # the 11 bands of Sentinel-3 in order.    
    band_names = ['F1','F2','S7','S8','S9','S1','S2','S3','S4','S5','S6']
    #
    # loading using ENVI
    if (data_type=='ENVI') or data_type=='envi':
        if verbose>0:
            print('--------------------------\nSentinel-3 image is loading\n')
        #preparing output    
        OUTPUT = []
        if verbose>0:
            if len(path)>1:
                print(str(len(path))+' images are being loaded')
            else:
                print(str(len(path))+' image is being loaded')
            
        #preparing all the lists.
        raster = []
        images = []
        extend = []
        names = []
        extra = []
        band_name=[]
        #loading each image is path, one at a time.
        for i in range(len(path)):
            if verbose>0:
                print('image '+str(i+1))
            
            #opening driver
            driver = gdal.GetDriverByName('ENVI') 
            driver.Register()
            #opening path
            dataset = gdal.Open(path[i], gdal.GA_ReadOnly)
            if dataset is None:
                #missing hdr
                sys.exit("Try again!")
            else:
                #number of bands, i.e F1, F2....
                bands = dataset.RasterCount
                if bands>1 and verbose>0:
                    print('There are '+str(bands)+' bands')
 
                if bands==1 and verbose>0:
                    print('There is '+str(bands)+' band')
                #loading each band
                for j in range(bands):
                    #getting names
                    names.append([path[i].split('\\')[-2:],band_names[j]])
                    #reading image
                    #img_temp = dataset.read(j+1)
                    #getting columns
                    cols = dataset.RasterXSize
                    #getting rows
                    rows = dataset.RasterYSize
                    # getting geo info.
                    xoffset, px_w, rot1, yoffset, px_h, rot2 = dataset.GetGeoTransform()
                    tryi = rotate_points_s3(xoffset-px_w*cols,yoffset+px_w*rows,rot2,xoffset, yoffset)
                    
                    if j==1:
                        new_x = math.cos(rot2)*(xoffset-px_w*cols) - math.sin(rot2)*(yoffset+px_w*rows)
                        new_y = math.sin(rot2)*(xoffset-px_w*cols) + math.cos(rot2)*(yoffset+px_w*rows)
                        
                    #finding coordinates of image
                    extend_temp = [xoffset-px_w*cols,xoffset,yoffset+px_w*rows,yoffset]
                    #getting raster info
                    band = dataset.GetRasterBand(j+1)
                    #getting image
                    img_temp = band.ReadAsArray(0, 0, cols, rows)
                    #finding no data valyes
                    NDV = dataset.GetRasterBand(1).GetNoDataValue()
                    #getting projections
                    projection = dataset.GetProjection()
                    stats = band.GetStatistics( True, True )
                    images.append(img_temp)
                    extend.append(extend_temp)
                    raster.append(band)               
                    extra.append([band,dataset,cols,rows,xoffset, px_w, rot1, yoffset, px_h, rot2,projection,stats])
                    del cols, rows, extend_temp, band, img_temp, NDV, projection
        del driver, dataset
                
                
    if data_type!='ENVI':
        assert data_type=='ENVI', 'this needs a bit of fixing. So I put in this catch statement.. sorry.'
        print('you sure? use ENVI.')
        OUTPUT = []
        print(str(len(path))+' images are being loaded')
        raster = []
        images = []
        extend = []
        names = []
        extra = []
        for i in range(len(path)):
            names.append(path[i].split('\\')[-2:]) 
        
        for i in range(len(path)):
            print('image '+str(i+1)+' is being loaded...')
            temp= rasterio.open(path[i], GEOREF_SOURCES='INTERNAL')
            bands = temp.count
            img_array = []
            mask = temp.dataset_mask()
            extra.append(mask)
            if bands>1:
                    print('There are '+str(bands)+' bands')
                    
            if bands==1:
                    print('There is '+str(bands)+' band')
                    
            for j in range(bands):               
                img_temp = temp.read(j+1)
                img_temp = np.clip(img_temp, np.quantile(img_temp, 0.1), np.quantile(img_temp, 0.9))
                img_temp = img_temp/(img_temp.max()/255.0)                
                images.append(np.flip(img_temp,2))
                img_extend= np.asarray(temp.bounds)[[0,2,1,3]]
                extend.append(img_extend)
                raster.append(temp) 

    
    return images,raster,extend, names, extra


def s3_show(img,im_extend=False,pic_name=False):
    '''
    Simply showing s1 image
    '''
    
    fig, axs = plt.subplots(figsize=(12,12), facecolor='w', edgecolor='k')        
    if any(im_extend):
        axs.imshow(img,cmap='binary', extent=im_extend)
    else:
        axs.imshow(img,cmap='binary')
        
    #axs.imshow(img[1][2000:6000,5000:10000],cmap='binary')
    #axs.axis('off')
    if any(pic_name):
        plt.title(str(pic_name[-1]))

    plt.xlabel('Longitude [deg]', fontsize=18)
    plt.ylabel('latitude [deg]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    return None



def rotate_points_s3(x,y, angle, offset_x,offset_y):
    """
    rotate x and y around x0 and y0..
        x is the found x distance from gdal
        y is the found y distance from gdal
        offset_x from gdal
        offset_y from gdal 
        angle is rot_2 from gdal..
    

    """
    adjusted_x = abs(x - offset_x)
    adjusted_y = abs(y - offset_y)
    cos_rad = math.cos(angle)
    sin_rad = math.sin(angle)
    qx = offset_x - cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y - -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy


#def get_img(path,data_type='ENVI',verbose=0):
#    """
#    s3.get_img():
#    
#    This fucntion loads the Sentinel-3 images into python using the gdal library.
#    The function takes all image bands in the path, and saves the raster information. the image as an array, and all the other information.
#    
#    Input:
#        path[str]: String of path of tif file.
#        data_type[str]: which driver shoud be used. Default and suggested is ENVi.
#        
#    Output:
#        images[list]: list of arrays. 
#        raster[obj]: object with raster information
#        extend[list]: list of geoprahic information. For e.g. plotting.
#        names[list]: list of string names.
#        extra[list]: list of advanced info. Here, offset, rotation and more is given.
#        
#    Example:
#        images,raster,extend, names, extra = s1.get_img('home/syntese/data/s3/s3_23.tif',data_type='ENVI')
#        
#    Author:
#        Kristian Aalling Soerensen
#        May 2020
#        kristian.sorensen@hotmail.com
#    """
#    # the 11 bands of Sentinel-3 in order.    
#    band_names = ['F1','F2','S7','S8','S9','S1','S2','S3','S4','S5','S6']
#    #
#    # loading using ENVI
#    if (data_type=='ENVI') or data_type=='envi':
#        if verbose>0:
#            print('--------------------------\nSentinel-3 image is loading\n')
#        #preparing output    
#        OUTPUT = []
#        if verbose>0:
#            if len(path)>1:
#                print(str(len(path))+' images are being loaded')
#            else:
#                print(str(len(path))+' image is being loaded')
#            
#        #preparing all the lists.
#        raster = []
#        images = []
#        extend = []
#        names = []
#        extra = []
#        band_name=[]
#        #loading each image is path, one at a time.
#        for i in range(len(path)):
#            if verbose>0:
#                print('image '+str(i+1))
#            
#            #opening driver
#            driver = gdal.GetDriverByName('ENVI') 
#            driver.Register()
#            #opening path
#            dataset = gdal.Open(path[i], gdal.GA_ReadOnly)
#            if dataset is None:
#                #missing hdr
#                sys.exit("Try again!")
#            else:
#                #number of bands, i.e F1, F2....
#                bands = dataset.RasterCount
#                if bands>1 and verbose>0:
#                    print('There are '+str(bands)+' bands')
# 
#                if bands==1 and verbose>0:
#                    print('There is '+str(bands)+' band')
#                #loading each band
#                for j in range(bands):
#                    #getting names
#                    names.append([path[i].split('\\')[-2:],band_names[j]])
#                    #reading image
#                    #img_temp = dataset.read(j+1)
#                    #getting columns
#                    cols = dataset.RasterXSize
#                    #getting rows
#                    rows = dataset.RasterYSize
#                    # getting geo info.
#                    xoffset, px_w, rot1, yoffset, px_h, rot2 = dataset.GetGeoTransform()
#                    tryi = rotate_points_s3(xoffset-px_w*cols,yoffset+px_w*rows,rot2,xoffset, yoffset)
#                    
#                    if j==1:
#                        new_x = math.cos(rot2)*(xoffset-px_w*cols) - math.sin(rot2)*(yoffset+px_w*rows)
#                        new_y = math.sin(rot2)*(xoffset-px_w*cols) + math.cos(rot2)*(yoffset+px_w*rows)
#                        #print('startvalues',xoffset,yoffset)
#                        #print('new',new_x,new_y)
#                        #print('function',tryi)
#                        #print('normal',xoffset-px_w*cols,yoffset+px_w*rows)
#                        
#                    #finding coordinates of image
#                    extend_temp = [xoffset-px_w*cols,xoffset,yoffset+px_w*rows,yoffset]
#                    
#                    #getting raster info
#                    band = dataset.GetRasterBand(j+1)
#                    #getting image
#                    img_temp = band.ReadAsArray(0, 0, cols, rows)
#                    #finding no data valyes
#                    NDV = dataset.GetRasterBand(1).GetNoDataValue()
#                    #getting projections
#                    projection = dataset.GetProjection()
#                    #making a BAD CLIPPING!!!!
#                    #img_temp = np.clip(img_temp, np.quantile(img_temp, 0.1), np.quantile(img_temp, 0.9))
#                    #img_temp = tools.img_stretch(img_temp)
#                    #img_temp = tools.img_normalize(img_temp)
#                    stats = band.GetStatistics( True, True )
#                    #img_temp = img_temp/(img_temp.max()/255.0)   
#                    #saving
#                    #images.append(np.flip(img_temp,1))
#                    images.append(img_temp)
#                    extend.append(extend_temp)
#                    raster.append(band)               
#                    extra.append([band,dataset,cols,rows,xoffset, px_w, rot1, yoffset, px_h, rot2,projection,stats])
#                    del cols, rows, extend_temp, band, img_temp, NDV, projection
#        del driver, dataset
#                
#                
#    if data_type!='ENVI':
#        print('TIF images')
#        OUTPUT = []
#        print(str(len(path))+' images are being loaded')
#        raster = []
#        images = []
#        extend = []
#        names = []
#        extra = []
#        for i in range(len(path)):
#            names.append(path[i].split('\\')[-2:]) 
#        
#        for i in range(len(path)):
#            print('image '+str(i+1)+' is being loaded...')
#            temp= rasterio.open(path[i], GEOREF_SOURCES='INTERNAL')
#            bands = temp.count
#            img_array = []
#            mask = temp.dataset_mask()
#            extra.append(mask)
#            if bands>1:
#                    print('There are '+str(bands)+' bands')
#                    
#            if bands==1:
#                    print('There is '+str(bands)+' band')
#                    
#            for j in range(bands):               
#                img_temp = temp.read(j+1)
#                img_temp = np.clip(img_temp, np.quantile(img_temp, 0.1), np.quantile(img_temp, 0.9))
#                img_temp = img_temp/(img_temp.max()/255.0)                
#                images.append(np.flip(img_temp,2))
#                img_extend= np.asarray(temp.bounds)[[0,2,1,3]]
#                extend.append(img_extend)
#                raster.append(temp) 
#            
#            #output = [images,raster,extend, names]
#            
#            #OUTPUT.append(output)
#    
#    return images,raster,extend, names, extra