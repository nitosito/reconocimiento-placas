
PLACA_COLOMBIA - v8 2025-06-08 10:37am
==============================

This dataset was exported via roboflow.com on June 8, 2025 at 4:16 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 3352 images.
Plate are annotated in YOLOv11 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)
* Auto-contrast via adaptive equalization

The following augmentation was applied to create 7 versions of each source image:
* Random rotation of between -15 and +15 degrees
* Random brigthness adjustment of between -25 and +25 percent
* Random exposure adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 4.6 pixels
* Salt and pepper noise was applied to 0.66 percent of pixels


