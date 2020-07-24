# Notebooks 
--------------------------------------------
```
s1_analysis_fires.ipynb
```
A short analysis of the Sentinel-1 GRD IW product's abilities to detect fires. The products is pre-processed using ESA SNAP. It is debatable weather or not it is possible to detect fires using it.
```
s1_analysis_past.ipynb
```
An initial analysis of the change in different images. Here, 5 images are acquired before the Ljusdal fire. Again, it is debatable if it is possible to detect a fire. 
```
s3_analysis_fires.ipynb
```
An analysis of the Sentinel-3 L1 RBT SLSTR product's ability to detect fires. Here, an analysis of all 11 bands, and the 5 bands used in the project is made. It is argued that it is possible to detect fires. This corresponds well with know fire products using this data.

```
modis_implementation.ipynb
```
This notebook illustrate show modis data is loaded and how the data is used to find the fires needed. One of the fires illustrated is the one used in the Sentinel-1 and Sentinel-3 analysis.

```
cnn_dataset.ipynb
```
How to load data, define areas with fires and no fires, and make arrays of these. These arrays are then saved as images that are used for labelling the data.

```
cnn_training.ipynb
```
This notebook shows how the CNNs are made. The arcitectures are defined using made functions. The models are then trained on the data made in cnn_dataset. 

```
cnn_testing.ipynb
```
The models capabilities are here tested. The cnn_training.ipynb shows how a model is found. These models are here used to detect fires with a certain probability on an entire image.

```
preparing_tutorial.ipynb
```
This notebook walks through the steps done in the preparation. It is meant as a summary. Few comments are made, and many functions are "hidded" withing the main functions. 


