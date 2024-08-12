# Image Similarity Search using Milvus and ResNet50



## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Step 1: Load the ResNet50 Model](#step-1-load-the-resnet50-model)
  - [Step 2: Connect to Milvus](#step-2-connect-to-milvus)
  - [Step 3: Define the Milvus Collection Schema](#step-3-define-the-milvus-collection-schema)
  - [Step 4: Extract Features](#step-4-extract-features)
  - [Step 5: Insert Features into Milvus](#step-5-insert-features-into-milvus)
  - [Step 6: Find Similar Images](#step-6-find-similar-images)
  - [Step 7: Display Similar Images](#step-7-display-similar-images)


## Introduction
This project demonstrates how to build an image similarity search system using Milvus, a powerful vector database, and ResNet50, a deep learning model for feature extraction. The code provides functions to insert image features into Milvus, create an index, and search for similar images based on a given query image.

## Prerequisites
- Python 3.x
- NumPy
- OpenCV
- Matplotlib
- Milvus
- TensorFlow

## Installation
1. Clone the repository: `git clone https://github.com/your-username/image-similarity-search.git`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage
1. Prepare your image dataset and place it in the appropriate directory.
2. Adjust the `image_folder` variable in the `main` function to point to your image directory.
3. Run the script: `python image_similarity_search.py`

The script will perform the following steps:

### Step 1: Load the ResNet50 Model
The ResNet50 model is loaded with pre-trained weights from ImageNet.

### Step 2: Connect to Milvus
The script connects to a Milvus server running on `localhost:19530`. Update the connection details if necessary.

### Step 3: Define the Milvus Collection Schema
The collection schema is defined with three fields: `_id` (primary key), `embedding` (feature vector), and `image_id` (optional image ID).

### Step 4: Extract Features
The `extract_features` function takes an image path and returns the feature vector using the ResNet50 model.

### Step 5: Insert Features into Milvus
The `insert_features` function iterates through the images in the specified folder, extracts features using `extract_features`, and inserts them into a Milvus collection. It also creates an index for efficient searching.

### Step 6: Find Similar Images
The `find_similar_images` function takes a user image path and returns the top-k most similar images based on the feature vector similarity.

### Step 7: Display Similar Images
The `display_similar_images` function displays the user image and the top-k most similar images along with their distances.



