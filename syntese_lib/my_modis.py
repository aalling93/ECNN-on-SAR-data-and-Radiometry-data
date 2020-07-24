import os
from glob import glob
import math
from matplotlib import pyplot as plt
import math
from rasterio import plot as rioplot
import geopandas as gpd
import gdal
import ogr
import osr
import gdalnumeric
import gdalconst
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from rasterio.plot import show
from rasterio import plot as rioplot
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from geojson import Point, Feature, FeatureCollection, dump



def modistoshape(path='',verbose=1,file_type='shp'):
    """
    Takes a path, finding all modis data in the dir and sub dir, 
    and load those.
    
    inputs
        path[str]: string of path. 
        verbose[int]: information. If verbose>0 info will be printed.
        file_type[str]: extention of files. Default is .shp files.
        
    output:
        file_paths[str]: table of path strings.
        
    Example:
        temp = modistoshape('C:\\Users\\krist\syntese\data',verbose=0)
        
    Author: 
        Kristian Aalling Soerensen
        May 2020
        kristian.sorensen@hotmail.com
    """
    allowed_types = ['shp']
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

def plotfire_shp(df_fire,title='fire'):
    """
        this fucntion plots the fires..
    """
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    #sweden = gpd.read_file('modis/Sweden.shp')
    #world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    Sweden = world.query('name == "Sweden"')
    Norway = world.query('name == "Norway"')
    Denmark = world.query('name == "Denmark"')
    
    ax = df_fire.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
    Sweden.plot(ax=ax, color='white', edgecolor='goldenrod',cmap='YlGn',alpha=1)
    Norway.plot(ax=ax, color='white', edgecolor='blue',cmap='Blues',alpha=1)
    Denmark.plot(ax=ax, color='white', edgecolor='red',cmap='Reds',alpha=1)
    df_fire.plot(column=df_fire.columns[2],
               ax=ax,
               legend=True,
               legend_kwds={'label': "Brightness",'orientation': "horizontal"},
                 cmap = 'YlOrRd')

    plt.xlim(5, 25)
    plt.ylim(55,70)
    plt.ylabel('latitude', fontsize=20)
    plt.xlabel('longitude', fontsize=20)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 15)
    
    plt.savefig(title+'.png', bbox_inches='tight')
    
    plt.show()
    
    
    return None

def get_fires(modis_data='shapefile',folder_name='geojson',verbose=0):
    """
        This function takes the fires in a shapefile, and turn it into geojson footprint.
        This footprint can then be used to download other day.
        
        Input:
            modis_data: a shafile with the fires, as loaded using modistoshape()
            folder_name: A string with the path for the files,
            verbose: how much info you wanna see.
        output:
            None. geojson files will be saved in folder
        example:
            get_fires(modis_data=shps[2],folder_name='geojson',verbose=0)
            
        Authro:
            Kristian Soerensen
            June 2020
            kristian.sorensen@hotmail.com
    """
    #all the fires
    data_FIRMS = gpd.read_file(modis_data)
    #only find fires with high confidence, high brightness and high confidence
    fires_used = data_FIRMS[(data_FIRMS[data_FIRMS.columns[2]]>260)  & (data_FIRMS[data_FIRMS.columns[11]]>250) & (data_FIRMS[data_FIRMS.columns[12]]>100) & (data_FIRMS[data_FIRMS.columns[9]]>90)]
    #creating a folder for .geojson files it it doesnt exitst.
    if not os.path.exists('geojson'):
        os.makedirs('geojson')
        
    #getting current dir, and changing dir to geojson
    cwd = os.getcwd()
    if verbose>0:
        print('------------------------------')
        print('(get_fires): Changing directory to '+str(cwd)+"/geojson")
        print('------------------------------')
        
    os.chdir(cwd+"/geojson")
    #now, these fires shoulw be made into geojson so it can be used to find sentinel 1 images. 
    temp_var = 0
    for i in range(len(fires_used)):
        # used to count..
        
        #making the name of the file
        name=str(fires_used['INSTRUMENT'].iloc[i])+'_'+str(fires_used['ACQ_DATE'].iloc[i])+'_'+str(fires_used['LATITUDE'].iloc[i])+'_'+str(fires_used['LONGITUDE'].iloc[i])+'.geojson'
        #only create the file if it doesnt exist...
        if os.path.exists(str(cwd)+"/geojson/"+name)==True:
            if verbose>0:
                print(str(i+1)+' (get_fires): file '+name+" already exists")
            
        if os.path.exists(str(cwd)+"/geojson"+name)==False: 
            temp_var = temp_var+1
            #getting coordiantes for point, used to make footprint
            latitude = fires_used['LATITUDE'].iloc[i]
            longitude = fires_used['LONGITUDE'].iloc[i]
            #getting the point for geojson
            point = Point((longitude,latitude))
            #its features(actually not needed for the foodprint)
            features = []
            features.append(Feature(geometry=point, properties={"BRIGHTNESS": str(fires_used['BRIGHTNESS'].iloc[i])}))
            features.append(Feature(geometry=point, properties={"INSTRUMENT": str(fires_used['INSTRUMENT'].iloc[i])}))
            features.append(Feature(geometry=point, properties={"ACQ_DATE": str(fires_used['ACQ_DATE'].iloc[i])}))
            features.append(Feature(geometry=point, properties={"FRP": str(fires_used['FRP'].iloc[i])}))
            features.append(Feature(geometry=point, properties={"CONFIDENCE": str(fires_used['CONFIDENCE'].iloc[i])}))
            features.append(Feature(geometry=point, properties={"BRIGHT_T31": str(fires_used['BRIGHT_T31'].iloc[i])}))
            features.append(Feature(geometry=point, properties={"ACQ_DATE": str(fires_used['ACQ_DATE'].iloc[i])}))

            feature_collection = FeatureCollection(features)
            
            
            #creating file
            with open(name, 'w') as f:
                dump(feature_collection, f)
    if verbose>0:
        if temp_var>0:
            print('(get_fires): '+str(temp_var+1)+' .geojson files are made')
        print('------------------------------')
        print('(get_fires): Changing directory back to '+str(cwd))
        print('------------------------------')            
    #going back to original dir
    os.chdir(cwd)
            
    return None   