# Implementing an Ensemble Convolutional Neural Network on Sentinel-1 Synthetic Aperture Radar data and Sentinel-3 Radiometric data for the detecting of forest fires
This Github repository serves as an extension for the report  "Implementing an Ensemble Convolutional Neural Network on
Sentinel-1 Synthetic Aperture Radar data and Sentinel-3 Radiometric data for the detecting of forest fires", made as a part of my MSc Eng. studies on Earth and Space Physcis and Engineering, with a specialisation in Earth Observation.

## Content
* Introduction
* Motivation with the final report
* Demo
* Libraries
* Setup
* How to use
* Credits




## Introduction:
- The ECNN automatically finds forest fires probabilities using geo-located Sentinel-1 SAR data and Sentinel-3 radiometry data.
- Focus of the project was not to get the best testing accuracy. Instead, the entire workflow should be developed(aquire data, process data, label data, define arcitecture, train models, get model results, test on new data, automate it all). 



An Ensemble Convolution Neural Network(ECNN) is a machine learning technique that can automatically extract features from arrays and use these features for classification problems. In an ECNN, it is possible to combine many type of data sets with the only requirement of them being comparable arrays. In this project, the Sentinel-1 SAR data and Sentinel-3 radiometric data has  be used to detect forest fires using such an ECNN. The entire framework of an ECNN is  described and implemented in Python using ESA SNAP to pre-process the data. It is argued that the ECNN was able to extract features that resembled the patterns of the fires. The framework is proved to work, and could be further improved or adjusted to other problems.


In short, the repository contains the entire implementation of an ECNN, starting with the acqusition of data, making the data set, labelling the data set, 
making an ECNN model, training the model and lastly using the model for new predictions.
Furthermore, an analysis of the Sentinel-1 IW GRD and Sentinel-3 L1 RBT SLSTR fire detection capabilities are made(a short one since focus is on the ECNN).

The framework works for the intended purposes. Many bugs exists, and problem occurs. Focus has not been on making the best product.

Images used, made training image and more is all shared in 
https://drive.google.com/drive/folders/1r4wy1NTS7uhgJvO991YsAXKmdUdMmYWG?usp=sharing





## Motivation
A growing need for advanced machine learning and deep learning algorithms is the motivating this project.
Here, the ECNN capabilities for Remote Sensing will be analyzed.
This is done in accordance with my studies, and the project is worth 10 ECTS points.

<object data="https://drive.google.com/file/d/1EqZN6q14--OdUyuOBrCkU-fHkgwq-x8B/view?usp=sharing" type="application/pdf" width="700px" height="700px">
    <embed src="https://drive.google.com/file/d/1EqZN6q14--OdUyuOBrCkU-fHkgwq-x8B/view?usp=sharing">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://drive.google.com/file/d/1EqZN6q14--OdUyuOBrCkU-fHkgwq-x8B/view?usp=sharing">Download PDF</a>.</p>
    </embed>
</object>

## Demonstration
Using the link above, images, models and implemneted modules can be downloaded, otherwise use gitclone.

To get a fire detection probability map, use the following function:
```python
prediction_map = cnn.get_prediction_one_image(folder_img='/content/drive/My Drive/syntese/data/cnn/images_testing/image_pair_1',
folder_models='/content/drive/My Drive/syntese/data/cnn/models/new',
verbose=1)
```
with the first input being the image and the second the trained model. With a verbose=1, the following will be printed:

![Printed](not_used/predictions.png?raw=true "Title")

And the resulting probability map is shown below:

![Printed](not_used/fire_detection_result.PNG?raw=true "Title")

We can **clearly** see a correlation with the actualy fire (shown to the left) with the predicted fire(shown in red to the right). Similary, we can see that e.g. clouds have a low probability of being a fire. This is remarkable considering only about 20 original subsets was used for training (they were aurgmented). -- This is not enough data, but still illustrated the possibilities.




## Libraries
```
my_cnn
```
Library used for the ECNN
```
my_s1
```
Loading and working with Sentinel-1 images

```
my_s3
```
Loading and working with Sentinel-3 images
```
my_modis
```
Loading and working with MODIS data. 

```
my_tools
```
General tools used in the project

### Notebooks
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

# Setup
-----------------------
```shell
$ git clone https://github.com/aalling93/syntese
```



# Credits
---------------------
This project is made by me, Kristian Aalling Sørensen, www.linkedin.com/in/ksoerensen. It is made in accordance to my specilisation on Earth Observation. 
The project is made in collaboration with Inge Sandholt and Kenan Vilic with Henning Skriver as the formal supervisor. 



