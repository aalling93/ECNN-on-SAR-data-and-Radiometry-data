# Implementing an Ensemble Convolutional Neural Network on Sentinel-1 Synthetic Aperture Radar data and Sentinel-3 Radiometric data for the detecting of forest fires
This Github repository serves as an extension for a report made  "Implementing an Ensemble Convolutional Neural Network on
Sentinel-1 Synthetic Aperture Radar data and Sentinel-3 Radiometric
data for the detecting of forest fires", made as a part of my studies on Earth and Space Physcis and Engineering, Earth Observation, MSc.

The abstract from the report is as follows:

Ensemble Convolution Neural Network(ECNN) is a machine learning technique that can
automatically extract features from arrays and use these features for classification problems.
In a ECNN, it is possible to combine many type of data sets with the only requirement
of them being arrays. In this project, the Sentinel-1 SAR data and Sentinel-3
radiometric data will be used to detect forest fires using such a ECNN. The entire framework
of a ECNN will be described and implemented in Python using ESA SNAP to pre
process the data. The resulting classification schema does not classify forest fires correctly,
with a training accuracy close to 100% and a testing accuracy around 50% equivalent to
random guessing in a two-class classification problem. It was argued that the ECNN was
able to extract features that resembled the patterns of the fires. The framework is proved
to work and could be further improved or adjusted to other problems.


In short, the repository contains the entire implementation of a ECNN, starting with the acqusition of data, making data-set, labelling data-set, 
making a ECNN model, training the model and lastly using the model for new predictions.
Furthermore, an analysis of the Sentinel-1 IW GRD and Sentinel-3 L1 RBT SLSTR fire detection capabilities are made(a short one since focus is on the ECNN).

The framework works for the intended purposes. Many bugs exists, and problem occurs. Focus has not been on making the best product.

# Content
----------------------------
* Motivation with the final report
* Build status
* Content of product
* Code style
* code example
* How to use
* Installation
* Author


# Motivation
----------------------------
I am currently doing a specialization on Earth Observation using mainly satellite data. 
A growing need for advanced machine learning algorithms is motivating this project.
Here, the ECNN capabilities for Remote Sensing will be analyzed.
This is done in accordance with my studies, and the project is consequently worth 10 ECTS points.


<object data="http://yoursite.com/the.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="http://yoursite.com/the.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://yoursite.com/the.pdf">Download PDF</a>.</p>
    </embed>
</object>


# Build status
------------------------------

[![Build Status](https://travis-ci.org/akashnimare/foco.svg?branch=master)](https://travis-ci.org/akashnimare/foco)
[![Windows Build Status](https://ci.appveyor.com/api/projects/status/github/akashnimare/foco?branch=master&svg=true)](https://ci.appveyor.com/project/akashnimare/foco/branch/master)


# Code style
------------------------------
