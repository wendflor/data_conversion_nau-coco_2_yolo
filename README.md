# data_conversion_nau-coco_2_yolo
## Table of Contents
1. [About the Project](#about-the-project)
2. [Installation](#installation)
3. [Usage](#usage)

## About the Project
This implementation allows converting datasets for **object detection** and **instance segmentation** from the **COCO format** (a widely used format for these tasks) into the **YOLO format**. This is particularly useful for utilizing datasets originally created for COCO-based models (e.g., Mask R-CNN) with YOLO-based models.
The project aims at the conversion of the output datasets of the work by Naumann et al., who expanded and adapted the COCO format, into the YOLO format.
Additionally, the project allows to merge two datasets with different objects (YOLO format). 

## Installation

To install the project locally, follow these steps:

Windows:
```bash
git clone https://github.com/wendflor/data_conversion_nau-coco_2_yolo.git
cd data_conversion_nau-coco_2_yolo
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Linux:
```bash
git clone https://github.com/wendflor/data_conversion_nau-coco_2_yolo.git
cd data_conversion_nau-coco_2_yolo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Usage
The original dataset consists of three subsets train, val and test. 
The general process is divided into a conversion and a merging process.
The conversion process consists of two main steps. 
In the first step the subsets are converted to general COCO format by executing the script convert_data_to_simple_coco.py.
The subsets are accessed by referencing the corresponding .json-filepaths in convert_data_to_simple_coco.py.
It is also necessary to define one general output folderpath.
In the second step the subsets expressed in generalized COCO format are converted to YOLO format by executing the script convert_coco_to_yolo_od_and_seg.py.
The subsets are accessed by referencing the corresponding .json-filepaths in convert_coco_to_yolo_od_and_seg.py.
It is also necessary to define one output folderpath for each subset.
The execution of the script merge_datasets.py merges the corresponding subsets of two datasets in YOLO format and with different objects into one common dataset with three subsets train, val and test. 
merge_datasets.py only needs the paths to the different dataset folders as well as an output folder path for saving the merged dataset.

If desired, it is possible to plot labels, masks and bboxes for each image to check and validate the groundtruth annotations.
This is done by create_groundtruth_visualization_box_and_masks.py. 
The script accesses the subsets of the merged dataset by the data.yaml file of the merged dataset.
