# Implementing an Ensemble Convolutional Neural Network on Sentinel-1 Synthetic Aperture Radar data and Sentinel-3 Radiometric data for the detecting of forest fires
This Github repository serves as an extension for a report  "Implementing an Ensemble Convolutional Neural Network on
Sentinel-1 Synthetic Aperture Radar data and Sentinel-3 Radiometric
data for the detecting of forest fires", made as a part of my studies on Earth and Space Physcis and Engineering, Earth Observation, MSc.

The abstract from the report is as follows:

An Ensemble Convolution Neural Network(ECNN) is a machine learning technique that can automatically extract features from arrays and use these features for classification problems. In an ECNN, it is possible to combine many type of data sets with the only requirement of them being arrays. In this project, the Sentinel-1 SAR data and Sentinel-3 radiometric data will be used to detect forest fires using such an ECNN. The entire framework of an ECNN will be described and implemented in Python using ESA SNAP to pre-process the data. The resulting classification schema does not classify forest fires correctly, with a training accuracy close to 100% and a testing accuracy around 50% equivalent to random guessing in a two-class classification problem. It is argued that the ECNN was able to extract features that resembled the patterns of the fires. The framework is proved to work, and could be further improved or adjusted to other problems.



In short, the repository contains the entire implementation of an ECNN, starting with the acqusition of data, making data-set, labelling data-set, 
making an ECNN model, training the model and lastly using the model for new predictions.
Furthermore, an analysis of the Sentinel-1 IW GRD and Sentinel-3 L1 RBT SLSTR fire detection capabilities are made(a short one since focus is on the ECNN).

The framework works for the intended purposes. Many bugs exists, and problem occurs. Focus has not been on making the best product.

Images used, made training image and more is all shared in 
https://drive.google.com/drive/folders/1r4wy1NTS7uhgJvO991YsAXKmdUdMmYWG?usp=sharing

# Content
----------------------------
* Motivation with the final report
* Libraries
* Setup
* How to use
* Credits



# Motivation
----------------------------
I am currently doing a specialization on Earth Observation using mainly satellite data. 
A growing need for advanced machine learning algorithms is motivating this project.
Here, the ECNN capabilities for Remote Sensing will be analyzed.
This is done in accordance with my studies, and the project is consequently worth 10 ECTS points.

<object data="https://drive.google.com/file/d/1EqZN6q14--OdUyuOBrCkU-fHkgwq-x8B/view?usp=sharing" type="application/pdf" width="700px" height="700px">
    <embed src="https://drive.google.com/file/d/1EqZN6q14--OdUyuOBrCkU-fHkgwq-x8B/view?usp=sharing">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://drive.google.com/file/d/1RqxQPLnNQpvNQYHj3ZtX7kfPIeRuL7Wj/view?usp=sharing">Download PDF</a>.</p>
    </embed>
</object>


# Libraries
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
my_tools
```
General tools used in the project

### Notebooks

See the notebook folder.

# Setup
-----------------------
```shell
$ git clone https://github.com/aalling93/syntese
```

# How to use
----------------------

![Detecting fires on images in folder](not_used/use1.png?raw=true "Title")

# Credits
---------------------
This project is made me, Kristian Aalling Sørensen, www.linkedin.com/in/ksoerensen. It is made in accordance to my studies on Earth Observation. 
The project is made in collaboration with Inge Sandholt and Kenan Vilic with Henning Skriver as the formal supervisor. 


