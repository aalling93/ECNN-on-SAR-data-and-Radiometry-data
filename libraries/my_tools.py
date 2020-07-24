import os
import numpy as np
import matplotlib.pyplot as plt
from os import path
import tifffile
from glob import glob
from matplotlib import pyplot as plt
import numpy as np


#from osgeo import gdal
#from osgeo import osr
#import gdal
#import ogr
#import osr
#import gdalnumeric
#import gdalconst
#import pandas as pd
#import rasterio as rio
#import geopandas as gpd
#from rasterio.plot import show
#from rasterio import plot as rioplot
#from imageio import imsave
#from PIL import Image
#import os.path
#import sys
#import math
#import rasterio # 
#import geopandas
#import contextily


## fra funktionen d. 21
import matplotlib
import os
from os import path
## fra funktionen d. 21
import matplotlib
import os
from os import path
def satf(data,name='',data_folder='C:\\Users\\krist\\Documents\\Syntese\\data\\NN_test',ext=".jpg"):
    '''
    save array to folder(satf)
    
    This function takes in defined arrays, and save the in their respective folders as images.
    The function is hard coded to make the images into data-sets of Sentinel-1 VV, Sentinel-1 VH, Sentinel-3 S13, Sentinel-3 S46 and Sentinel-3 S79. 
    For information on these combinations, see the Report.
    The arrays are saved into 2D and 3D arrays respectivly. The image type of decided by the input 'ext'.
    
    Each data-set will be saved into their own folder. For each folder, a sub-folder is made for each class.
    Each image will then be given a name depending on the type and id, and the 'name' input.
    For the Sentinel-1 VV data, the folder structure would be:
    data_folder/vv/fire/fire_1_name.jpg
    data_folder/vv/no_fire/no_fire_1_name.jpg
    
    
    Input:
        data[list]: List of arrays in the format [fire_box_s1,no_fire_box_s1,fire_box_s3,no_fire_box_s3]
        name[str]: specific name for data, could be date.
        data_folder[str]: String of folder to save all data-sets to.
    Output:
        None. Images will be saved.
        
    Example:
        satf([fire_box_s1,no_fire_box_s1,fire_box_s3,no_fire_box_s3],name='ljusdal_23_07_2018',data_folder='C:\\Users\\krist\\Documents\\Syntese\\data\\NN_test',ext=".jpg")
        
    Author:
        Kristian Aalling Soerensen
        July 2020
        kristian.sorensen@hotmail.com    
    '''
    fire_box_s1=data[0]
    no_fire_box_s1=data[1]
    fire_box_s3=data[2]
    no_fire_box_s3=data[3]
    
    s1_vh_fire  = []
    s1_vv_fire  = []
    s13_fire = []
    s46_fire = []
    s79_fire = []
    for i in range(len(fire_box_s1)):
        s1_vh_fire.append(fire_box_s1[i][:,:,0])
        s1_vv_fire.append(fire_box_s1[i][:,:,1])
        s13_fire.append(np.stack(([fire_box_s3[i][:,:,3],fire_box_s3[i][:,:,0],fire_box_s3[i][:,:,1]]), axis=2))
        s46_fire.append(np.stack(([fire_box_s3[i][:,:,4],fire_box_s3[i][:,:,0],fire_box_s3[i][:,:,1]]), axis=2))
        s79_fire.append(np.stack(([fire_box_s3[i][:,:,4],fire_box_s3[i][:,:,2],fire_box_s3[i][:,:,1]]), axis=2))
          
        
    s1_vh_no_fire  = []
    s1_vv_no_fire  = []
    s13_no_fire = []
    s46_no_fire = []
    s79_no_fire = []
    for i in range(len(no_fire_box_s1)):
        s1_vh_no_fire.append(no_fire_box_s1[i][:,:,0])
        s1_vv_no_fire.append(no_fire_box_s1[i][:,:,1])
        s13_no_fire.append(np.stack(([no_fire_box_s3[i][:,:,3],no_fire_box_s3[i][:,:,0],no_fire_box_s3[i][:,:,1]]), axis=2))
        s46_no_fire.append(np.stack(([no_fire_box_s3[i][:,:,4],no_fire_box_s3[i][:,:,0],no_fire_box_s3[i][:,:,1]]), axis=2))
        s79_no_fire.append(np.stack(([no_fire_box_s3[i][:,:,4],no_fire_box_s3[i][:,:,2],no_fire_box_s3[i][:,:,1]]), axis=2))
    
    img_list = [s1_vv_fire,s1_vv_no_fire,s1_vh_fire,s1_vh_no_fire,s13_fire,s13_no_fire,s46_fire,s46_no_fire,s79_fire,s79_no_fire]
    
    #folders = ['sentinel1\\vv\\fire','sentinel1\\vv\\no_fire','sentinel1\\vh\\fire','sentinel1\\vh\\no_fire','sentinel3\\13\\fire','sentinel3\\13\\no_fire','sentinel3\\46\\fire','sentinel3\\46\\no_fire','sentinel3\\79\\fire','sentinel3\\79\\no_fire']
    folders = ['vv\\fire','vv\\no_fire','vh\\fire','vh\\no_fire','13\\fire','13\\no_fire','46\\fire','46\\no_fire','79\\fire','79\\no_fire']
    
    for i in range(len(folders)):
        if os.path.exists(data_folder+'\\'+folders[i])==False:
            #os.mkdir(data_folder+'\\'+folders[i]) #ingen anelse om hvorfor den ikke virker...
            os.system("mkdir " + data_folder+'\\'+folders[i])
        
    
    for j in range(len(img_list)):
        if (j % 2) == 0:                
            for i in range(len(img_list[j])):
                if (folders[j][0:2]=='13') or (folders[j][0:2]=='46') or (folders[j][0:2]=='79'):
                    for l in range(3):
                        img_list[j][i][:,:,l] = ((img_list[j][i][:,:,l] -img_list[j][i][:,:,l].min())/(img_list[j][i][:,:,l].max()-img_list[j][i][:,:,l].min() )*1)
                
                img_name = data_folder+'\\'+folders[j]+'\\fire_'+str(i+1)+name+ext
                if path.exists(img_name)==True:
                    print('Files already exists')
                else:
                    matplotlib.image.imsave(img_name, img_list[j][i])
                    
        else:
            for i in range(len(img_list[j])):
                if folders[j][0:2]=='13' or folders[j][0:2]=='46' or folders[j][0:2]=='79':
                    for l in range(3):
                        img_list[j][i][:,:,l] = ((img_list[j][i][:,:,l] -img_list[j][i][:,:,l].min())/(img_list[j][i][:,:,l].max()-img_list[j][i][:,:,l].min() )*1)
                    
                img_name = data_folder+'\\'+folders[j]+'\\no_fire_'+str(i+1)+name+ext    
                if path.exists(img_name)==True:
                    print('Files already exists')
                else:
                    matplotlib.image.imsave(img_name, img_list[j][i])
                    
            

        
    
    return None





def img_stretch(img,Min=0,Max=1):
    '''
    This function stretched an image between the Min and Max value. 
    
    Input:
        img[array]: NxM image
        Min[float]: Smallest value in new iamge
        Max[float]: Biggest value in new image
    
    Output:
        img_stretched[array]: NxM array with new values
    
    Example:
        img_new = img_stretch(img,Min=0,Max=1)
    
    Author:
        Kristian Soerensen
        May 2020
        kristian.sorensen@hotmail.com
    '''
    
    #img = img.astype('float64') 
    v_max = img.max()
    v_min = img.min()
    
    img_stretched = (((img-v_min)/(v_max-v_min))*1)
    #.astype('uint8')
    #img_stretched = ((img-v_min)/(v_max-v_min))
    #img_stretched[img_stretched < Min] = Min
    #img_stretched[img_stretched > Max] = Max
    
    del v_max, v_min
    
    return img_stretched


def img_normalize(img):
    '''
    '''
    v_max = img.max()
        
    img_norm = img.astype(np.float64) / v_max # normalize the data to 0 - 1
    img_norm = img_norm *255
    img_norm = img_norm.astype(np.uint8)

    
    del v_max 
    
    return img_norm

def color_img(r,g,b):
    """
    color color representation.
    just illustration.. 
    (NxMx3) -> (NxMx1)
    """
    
    #color_image = np.stack((r*100,(g*42), (b*10)), axis=2)
    color_image = np.stack((r,g, b), axis=2)
    color_image = color_image/np.quantile(color_image,0.99)
    
    return color_image

def file_paths(path='',verbose=1,file_type='tif'):
    """
    Takes a path, finding all sentinel 1 images in the dir and sub dir, 
    and load those pics into python using gdal..
    
    
    inputs
        path[str]: string of path. 
        verbose[int]: information. If verbose>0 info will be printed.
        file_type[str]: extention of files. Default is .tif files.
        
    output:
        file_paths[str]: table of path strings.
        
    Example:
        temp = s1toimg('C:\\Users\\krist\\s1',verbose=0,file_type='tif')
        temp = s1toimg('C:\\Users\\krist\\s1',verbose=1,file_type='tif')
        temp = s1toimg('C:\\Users\\krist\\s1',verbose=2,file_type='tif')   
        
    Author: 
        Kristian Aalling Soerensen
        May 2020
        kristian.sorensen@hotmail.com
    """
    allowed_types = ['tif','png','jp2','dim','img','geojson','xml','SEN3','zip','SAFE','TIF','']
    assert (file_type in allowed_types),('\nERROR: '+file_type+': Illegal file type.')
    
    #getting current dir.
    cwd = os.getcwd()
    file_paths = []
    
    if path is '':
        start_dir = os.getcwd()
    
    if path is not '':
        start_dir = path
        if verbose>1:
            print('\nChangin dir to: '+start_dir)

    
    
    extention   = "*."+file_type
    for dir,_,_ in os.walk(start_dir):
        file_paths.extend(glob(os.path.join(dir,extention))) 
    
    assert len(file_paths)>0, ('\nERROR: No files of type:'+file_type+' in dir.')
    
    if verbose>0:
        print('\nthere are:',len(file_paths),' .',file_type,' files in folder')
    if verbose>1:
        print(file_paths)
    
    os.chdir(cwd)
    if verbose>1:
        print('\n Changin dir to: '+cwd)
        
    return file_paths



def run_gpt_model(image_pair_path,graph_folder):
    '''
    
    This script assumes you have ESA SNAP and its gpt extention installed and have it in your path.
    The function finds the Sentinel-1 and Sentinel-3 image pair. 
    It then runs the neccesary SNAP graphs on them (See the 'Implementing an Ensemble Convolutional Neural Network on
    Sentinel-1 Synthetic Aperture Radar data and Sentinel-3 Radiometric
    data for the detecting of forest fires' report)
    It then saves the processed images to the folder.
    
    Note. 
        To ensure that it workd, the graphs are hardcoded to the specific images! This can be generalized by 'easily' adding an input for the graph, i.e.
        input/output and target names. The function could then be changed to:
        run_gpt_model(image_pair_path,graph_folder,s3_target_name,s1_targetname,s2_wkt_file)
        where the wkt file should be a file describing the geometric subset...
    Note2:
        Some kind of error occured when testing. the collocation between S1 and S3 bugged. Step forum is one it. 
        Currecntly, the function does not work.
    
    input:
        image_pair_path[str]: path for folder witl Sentinel-1[zip] and Sentinel-1[SEN3] files.
        graph_folder[str]: path for the 3 graphs used: sentinel_1_preprocess.xml, sentine_3_preprocess.xml and s1s3_collocate.xml.        
    output:
        None: images will be saved
    example:
        run_gpt_model('user\syntese\images\s1s3folder','user\syntese\graph\project_graphs')
    
    Author:
        Kristian Aalling Soerensen
        June 2020
        Updated: July 2020
        Kristian.sorensen@hotmail.com
    '''
    assert (sys.platform=='win32'),'this is for windows'
    assert (len(image_pair_path)==2),'only have one image pair in folder.'
    
    #Sentinel-1 image is of type zip
    s1_path = tools.file_paths(image_pair_path,verbose=0,file_type='zip')[0]
    # Sentinel-3 image is of type SEN3 (should thus be unzipped)
    s3_path= tools.file_paths(image_pair_path,verbose=0,file_type='SEN3')[0]
    if s1_path==False:
        print('no s1 image')
    elif s3_path==False:
        print('no s3 image')

    
    #Getting the graphs needed. They are in the use_network folder and have a specific structure.
    graphs = tools.file_paths(graph_folder,verbose=0,file_type='xml')  
    #Graph to process sentinel-1
    graph_process_s1 = graphs[0]
    #Graph to process Sentinel-3
    graph_process_s3 = graphs[1]
    #Graph to collocate them and aquire the tif images.
    graph_collocate  = graphs[2]   

    cwd = os.getcwd()
    #making a folder to put processes images into
    os.mkdir(image_pair_path+'\\processed') 
    #------------------------------------------------
    # ----------- processing Sentinel-1 ------------
    #------------------------------------------------
    #preparing file
    targetfile_s1 = image_pair_path+'\\processed\\'+'S1'
    # defining gpt command
    command_gpt_s1 ='gpt "'+graph_process_s1+'" -PtargetFile='+targetfile+' "'+s1_path+'"'
    # defining cmd command (gpt is run through cmd on windows)
    command_cmd_s1 = 'cmd /k "'+command_gpt+'"'
    # run cmd command
    os.system(command_cmd_s1)
    #------------------------------------------------
    # ----------- processing Sentinel-3 ------------
    #------------------------------------------------
    #preparing file
    targetfile_s3 = image_pair_path+'\\processed\\'+'S3'
    # defining gpt command
    command_gpt_s3 ='gpt "'+graph_process_s3+'" -PtargetFile='+targetfile+' "'+s3_path+'"'
    # defining cmd command (gpt is run through cmd on windows)
    command_cmd_s3 = 'cmd /k "'+command_gpt+'"'
    # run cmd command
    os.system(command_cmd_s3)
    #------------------------------------------------
    # --------- collocating Sentinel pair------------
    #------------------------------------------------
    #preparing file
    targetfile_s1s3 = image_pair_path+'\\processed\\'+'S1'
    # defining gpt command
    command_gpt_s1s3 ='gpt "'+graph_collocate+'" -PtargetFile='+targetfile+' "'+s1_path+'"'+' "'+s3_path+'"'
    # defining cmd command (gpt is run through cmd on windows)
    command_cmd_s1s3 = 'cmd /k "'+command_gpt+'"'
    # run cmd command
    os.system(command_cmd_s1s3)
    
    
    
    os.chdir(cwd)
    return None