import os
import sys
import math
import numpy as np
import rasterio
import gdal
import matplotlib.pyplot as plt

from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
from matplotlib import pyplot as plt
from osgeo import gdal


#import my_tools as tools
#import rasterio as rio
#import geopandas as gpd
#from rasterio.plot import show
#from rasterio import plot as rioplot
#import ogr
#import osr
#import gdalnumeric
#import gdalconst
#import pandas as pd
#import geopandas 
#import contextily 
#from glob import glob
#from osgeo import osr
#import pickle
#import shutil
#import pandas as pd

def s1_show(img,im_extend=False,pic_name=False):
    '''
    Simply showing s1 image... nothing fancy.
    '''
    
    fig, axs = plt.subplots(figsize=(12,12), facecolor='w', edgecolor='k')        
    if any(im_extend):
        axs.imshow(img,cmap='binary', extent=im_extend)
    else:
        axs.imshow(img,cmap='binary')
        
    #axs.imshow(img[1][2000:6000,5000:10000],cmap='binary')
    #axs.axis('off')
    if any(pic_name):
        plt.title(str(pic_name))

    plt.xlabel('Longitude [deg]', fontsize=18)
    plt.ylabel('latitude [deg]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    return None

def download_sentinel1(footprint='C:\\Users\\krist\\Documents\\Syntese\\code\\notebooks\\ljundal.geojson',folder = 'C:\\Users\\krist\\Documents\\Syntese\\data\\s1\\download',username='aalling93',password= 'ntm22xrm',date_start=['01','06','2020'],date_end=['14','06','2020']):
    '''
        The following function downloads sentinel-1 data if it happens to be online. If it is online, it innitiates the LTR.
        The function needs a footprint. This foorprint can be made directly using e.g. modis.get_fires() function. Here, a foodprint for each fire has been made.
        This function also gives the date of the fire, which can be used.
        The function then download all Sentinel-1 images within that footprint, defined by the start and end data.
        
        The function right now donwload the data, and can initiate LRT. A later version should include automatic download after LTR.  
        
        more options can be added in the api, see https://pypi.org/project/sentinelsat/.
        Note, from Copernicus hub,  only two images can be downloaded at the same time. This is therefor run one at a time in a loop.
        ...
        
        Input:
            footprint[str]: A string with the path to a footprint in geojson format.
            folder[str]: The path to which the downloaded images will be saved. 
            username[str]: Copernicus Hub username
            password[str]: Copernicus Hub password
            date_start[list]: a list with the date in string format DAY-MONTH-YEAR (with 0 if e.g. 01)
            date_end[list]: a list with data in strin format DAY-MONTH-YEAR (eithout 0 if e.g. 01)
        
        output:
            NIL: No output. images will be donwloaded.
        
        Example
        
        Author:
            Kristian Soerensen
            July 2020
            kristian.sorensen@hotmail.com
        
    '''
    #initializing API
    api = SentinelAPI(username, password, 'https://scihub.copernicus.eu/dhus')
    Data_start=(str(date_start[2])+str(date_start[1])+str(date_start[0]))
    #fetching all product
    Footprint = geojson_to_wkt(read_geojson(footprint))
    #Footprint = footprint
    products = api.query(Footprint,date=(Data_start, date(int(date_end[2]), int(date_end[1]), int(date_end[0]))),producttype='GRD',orbitdirection='DESCENDING')
    
    cwd = os.getcwd()
    os.chdir(folder)
    #turning into dataframe
    products_df = api.to_dataframe(products)
    print(len(products_df),' Products are found.')
    #checking if file in folder
    for i in range(len(products_df)):
        if os.path.isfile(products_df.iloc[i].filename)==True:
            #drop the file if it is in the folder
            products_df.drop(products_df.index[i])
            print('(download_sentinel1)\n Product:',products_df.iloc[i].title,' Already exists.')
    #download all products in pd, 1 at a time.    
    for i in range(len(products_df)):
        api.download(products_df.iloc[i].uuid)
        
    os.chdir(cwd)
    return None

def get_img(path,data_type='ENVI'):
    """
    s1.get_img():
    
    This fucntion loads the Sentinel-1 images into python using the gdal library.
    The function takes all images in the path, and saves the raster information. the image as an array, and all the other information.
    
    Input:
        path[str]: String of path of tif file.
        data_type[str]: which driver shoud be used. Default and suggested is ENVi.
        
    Output:
        images[list]: list of arrays. 
        raster[obj]: object with raster information
        extend[list]: list of geoprahic information. For e.g. plotting.
        names[list]: list of string names.
        extra[list]: list of advanced info. Here, offset, rotation and more is given.
        
    Example:
        images,raster,extend, names, extra = s1.get_img('home/syntese/data/s1/s1_23.tif',data_type='ENVI')
        
    Author:
        Kristian Aalling Soerensen
        May 2020
        kristian.sorensen@hotmail.com
    """
    
    
    if (data_type=='ENVI') or data_type=='envi':
        print('--------------------------\nSentinel-1 image is loading\n')
        OUTPUT = []
        if len(path)>1:
            print(str(len(path))+' images are being loaded')
        else:
            print(str(len(path))+' image is being loaded')
            
        
        raster = []
        images = []
        extend = []
        names = []
        extra = []        
        for i in range(len(path)):
            print('image '+str(i+1))
            #opening driver
            driver = gdal.GetDriverByName('ENVI') 
            driver.Register()
            dataset = gdal.Open(path[i], gdal.GA_ReadOnly)
            if dataset is None:
                #missing hdr
                sys.exit("Try again!")
            else:
                #number of bands, i.e vv and vh
                bands = dataset.RasterCount
                if bands>1:
                    print('There are '+str(bands)+' bands')
 
                if bands==1:
                    print('There is '+str(bands)+' band')
                
                for j in range(bands):
                    #getting names
                    names.append(path[i].split('\\')[-2:])
                    #reading image
                    #img_temp = dataset.read(j+1)
                    #getting columns
                    cols = dataset.RasterXSize
                    #getting rows
                    rows = dataset.RasterYSize
                    # getting geo info.
                    xoffset, px_w, rot1, yoffset, px_h, rot2 = dataset.GetGeoTransform()
                    #finding coordinates of image
                    extend_temp = [xoffset-px_w*cols,xoffset,yoffset-px_w*rows,yoffset]
                    #getting raster info
                    band = dataset.GetRasterBand(j+1)
                    #getting image
                    img_temp = band.ReadAsArray(0, 0, cols, rows)
                    #finding no data valyes
                    NDV = dataset.GetRasterBand(1).GetNoDataValue()
                    #getting projections
                    projection = dataset.GetProjection()
                    if img_temp is not None:
                        #img_temp = tools.img_stretch(img_temp)
                        #making a BAD CLIPPING!!!!
                        #img_temp = np.clip(img_temp, np.quantile(img_temp, 0.01), np.quantile(img_temp, 0.99))
                        #img_temp = tools.img_normalize(img_temp)
                        #img_temp = img_temp/(img_temp.max()/255.0)   
                        #saving
                        #images.append(np.flip(img_temp,1))
                        images.append(img_temp)
                        extend.append(extend_temp)
                        raster.append(band)               
                        extra.append([band,dataset,cols,rows,xoffset, px_w, rot1, yoffset, px_h, rot2,projection ])
                    del cols, rows, extend_temp, band, img_temp, NDV, projection
        del driver, dataset
                
                
    if data_type!='ENVI':
        print('TIF images')
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
                images.append(np.flip(img_temp,1))
                img_extend= np.asarray(temp.bounds)[[0,2,1,3]]
                extend.append(img_extend)
                raster.append(temp) 
            
            #output = [images,raster,extend, names]
            
            #OUTPUT.append(output)
    
    return images,raster,extend, names, extra

def get_subsets(img):
    """
        The following function takes in images, and split them up in subsets of same size.
        
        There is an overlap in the subsets. 
        
        input:
            img[array]: table of numpy images.
            
        output:
            subsets_image[array]: A table containing all the subsets of size 800.
            
        Example:
            subsets = get_subsets(list_of_img)
                gets subsets for a list of many images.
            subsets = get_subsets(list_of_img[1])
                get subsets from the second image in a list.
            
        Author:
            Kristian Aalling Soerensen
            10. May 2020
            kristian.sorensen@hotmail.com
    """
    subsets_image = []
    #the size of the subsets are hardcoded since this value should be the same and never changed!
    size = 200
    half = int(math.ceil(size/2))
    for i in range(len(img)):
        rows, colums = img.shape
        subsets = [];          
        for i in range(half,colums-half,half):
            for j in range(half,rows-half,half):
                temp = img[i-half:i+half,j-half:j+half];
                subsets.append(temp)
                
        subsets_image.append(subsets)
    # NOTE!! NAAR JEG FAAR DATA, SKAL DE LABLES!!
    # np.stack((r,g, b), axis=2)
    return subsets_image 


#def rasterio_img(path,data_type):
#    """
#    OBSOLOTE
#    """
#
#    
#    if datatype.capitalize()=='ENVI':
#        driver = gdal.GetDriverByName('ENVI') 
#        driver.Register()
#        inDs = gdal.Open(path[i], gdal.GA_ReadOnly)
#        if inDs is None:
#            #missing hdr
#            sys.exit("Try again!")
#        else:
#            cols = inDs.RasterXSize
#            rows = inDs.RasterYSize
#            bands = inDs.RasterCount
#            
#            geotransform = inDs.GetGeoTransform()
#            geotransform = inDs.GetGeoTransform()
#            originX = geotransform[0]
#            originY = geotransform[3]
#            pixelWidth = geotransform[1]
#            pixelHeight = geotransform[5]
#            band = inDs.GetRasterBand(1)
#            image_array = band.ReadAsArray(0, 0, cols, rows)
#            image_array_name = file_name
#            
#            print (type(image_array))
#            OUTPUT = [image_array, pixelWidth, (geotransform, inDs)]
#    else:
#        raster = []
#        images = []
#        extend = []
#        name =[]
#        for i in range(len(path)):
#            temp= rasterio.open(path[i])
#            img_array = temp.read(1)
#            #img_array = np.clip(img_array, np.quantile(img_array, 0.1), np.quantile(img_array, 0.9), out=img_array)
#            img_array = np.clip(img_array, np.quantile(img_array, 0.1), np.quantile(img_array, 0.9))
#            #img_array = img_array/(img_array.max()/255.0)
#            images.append(img_array)
#            raster.append(temp)
#            img_extend= np.asarray(temp.bounds)[[0,2,1,3]]
#            extend.append(img_extend)
#            #pic_name= (path).name[-49:-38]
#            #name.append(pic_name)
#            
#            OUTPUT = [images,raster,extend, name]
#        
#    
#    return OUTPUT  
#
#
#def get_images(path):
#    """
#    OBSOLOTE
#        This function loads all images defined by path.
#        For each path, an image and raster data will be saved.
#        
#        input:
#            path[str]: a table containing strings
#        
#        output:
#            images[array]: A table containing all the images in numpy format
#            rasters[raster]: A table containing all the raster data.
#        
#        Example:
#            images, raster = get_images(path)
#            images, raster = get_images(path[0])
#            images, raster = get_images(path[2])
#            
#        Author:
#            Kristian Aalling Soerensen
#            10. May 2020
#            kristian.sorensen@hotmail.com
#    """
#    
#    images = []
#    rasters =[]
#    for i in range(len(path)):
#        raster = gdal.Open(path[i])
#        rasters.append(raster)
#        rb = raster.GetRasterBand(1)
#        img_array = rb.ReadAsArray()
#        print('a')
#        #img_array = np.clip(img_array, np.quantile(img_array, 0.1), np.quantile(img_array, 0.9), out=img_array)
#        img_array = np.clip(img_array, np.quantile(img_array, 0.1), np.quantile(img_array, 0.9))
#        img_array = img_array/(img_array.max()/255.0)
#        images.append(img_array)
#        
#    return images, rasters


#def plot_images(images,path=''):
#    """
#        function plot sar images
#        
#    """
#    
#    fig, axs = plt.subplots(math.ceil(len(images)/2),2, figsize=(12,12), facecolor='w', edgecolor='k')
#    fig.subplots_adjust(hspace = .5, wspace=.001)
#    axs = axs.ravel()
#    
#    for i in range(len(images)):
#        axs[i].imshow(images[i],cmap='binary')
#        #axs[i].imshow(images[i],cmap='RdYlBu')
#        axs[i].axis('off')
#        if path!='':
#            axs[i].set_title(path[i][17:21]+'-'+path[i][21:23]+'-'+path[i][23:25])
#        
#    return None

#def get_raster_info(raster):
#    """
#        takes raster
#    """
#    projections = []
#    meta_data = []
#    number_bands = []
#    dimensions = []
#    
#    for i in range(len(raster)):
#        projections.append(raster[i].GetProjection())
#        meta_data.append(raster[i].GetMetadata())
#        number_bands.appenmd(raster[i].RasterCount)
#        x = raster[i].RasterXSize
#        y = raster[i].RasterYSize
#        dimensions.append([x,y])
#        
#        
#    return projections, meta_data


#def get_statistics(raster):
#    """
#        get stats
#    """
#    stats = []
#    
#    for i in range(len(raster)):
#        band = raster[i].GetRasterBand(1)
#        no_data = band.GetNoDataValue()
#        minimum = band.GetMinimum()
#        maximum = band.GetMaximum()
#        stats.append([no_data,minimu,maksimum])
#        
#    return stats
#    
#
#
#from osgeo import gdal, gdalconst 
#from osgeo.gdalconst import *
#
#def adsf():
#    
#    if datatype.capitalize()=='ENVI':
#        driver = gdal.GetDriverByName('ENVI') 
#        driver.Register()
#        inDs = gdal.Open(file_name, GA_ReadOnly)
#        if inDs is None:
#            #missing hdr
#            sys.exit("Try again!")
#        else:
#            cols = inDs.RasterXSize
#            rows = inDs.RasterYSize
#            bands = inDs.RasterCount
#            
#            geotransform = inDs.GetGeoTransform()
#            geotransform = inDs.GetGeoTransform()
#            originX = geotransform[0]
#            originY = geotransform[3]
#            pixelWidth = geotransform[1]
#            pixelHeight = geotransform[5]
#            band = inDs.GetRasterBand(1)
#            image_array = band.ReadAsArray(0, 0, cols, rows)
#            image_array_name = file_name
#            
#            print (type(image_array))
#        
#    return image_array, pixelWidth, (geotransform, inDs)
        
        


