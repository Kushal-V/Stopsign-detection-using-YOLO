#!/usr/bin/env python
# coding: utf-8

# In[7]:


import json
import cv2
import os

with open(r"C:\Users\ASUS\Downloads\Stop-Sign.v3i.coco\valid\_annotations.coco.json", "r") as json_file:
    data = json.load(json_file)


# In[8]:


coco_to_yolo = {
    1: 0,  # Map COCO category ID to YOLO class index
}


# In[31]:


import json
import os

# Modify this dictionary to map your COCO class IDs to YOLO class indices
coco_to_yolo = {
    1: 0,  # Map COCO class ID 1 to YOLO class index 0
    2: 1,  # Map COCO class ID 2 to YOLO class index 1
    # Add more mappings for your specific classes
}

# Path to your COCO format JSON file
coco_json_path = r"C:\Users\ASUS\Downloads\Stop-Sign.v3i.coco\valid\_annotations.coco.json"

# Output directory for YOLO format text files
output_dir = 'yolo_annotations'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the COCO JSON data
with open(coco_json_path, 'r') as json_file:
    coco_data = json.load(json_file)

# Process each image and its annotations
for image in coco_data['images']:
    image_id = image['id']
    file_name = image['file_name']

    yolo_annotations = []
    for annotation in coco_data['annotations']:
        if annotation['image_id'] == image_id:
            category_id = annotation['category_id']
            if category_id in coco_to_yolo:
                bbox = annotation['bbox']
                x, y, width, height = bbox
                x_center = x + width / 2
                y_center = y + height / 2

                # Convert COCO class ID to YOLO class index
                class_index = coco_to_yolo[category_id]

                # Normalize the coordinates and dimensions
                x_center /= image['width']
                y_center /= image['height']
                width /= image['width']
                height /= image['height']

                yolo_annotations.append(f"{class_index} {x_center} {y_center} {width} {height}")

    # Write YOLO format text file for the image
    txt_file_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.txt')
    with open(txt_file_path, 'w') as txt_file:
        txt_file.write('\n'.join(yolo_annotations))

print(f"Converted {len(coco_data['images'])} images to YOLO format. Text files are saved in '{output_dir}' directory.")


# In[32]:


import os
import glob
from PIL import Image
import numpy as np

valid_image_dir = 'C:/Users/ASUS/Downloads/Stop-Sign.v3i.coco/valid/images/'
valid_label_dir = 'C:/Users/ASUS/Downloads/Stop-Sign.v3i.coco/valid/labels/'

image_files = sorted(glob.glob(os.path.join(valid_image_dir, '*.jpg')))
label_files = sorted(glob.glob(os.path.join(valid_label_dir, '*.txt')))

# Sample code to load and process images and labels
for image_file, label_file in zip(image_files, label_files):
    image = Image.open(image_file)
    image = np.array(image)  # Convert to NumPy array
    # Process the image as needed

    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            annotation = line.strip().split()
            class_id = int(annotation[0])
            x_center, y_center, width, height = map(float, annotation[1:])
            # Process the class ID, bounding box coordinates, etc.


# In[42]:


import os
import random
import shutil

# Define the paths to your original image and annotation folders
original_image_dir = r"C:\Users\ASUS\Downloads\Stop-Sign.v3i.coco\valid\images"
original_annotation_dir = r"C:\Users\ASUS\Downloads\Stop-Sign.v3i.coco\valid\labels"

# Define the paths for the training and validation sets
train_image_dir = r"C:\Users\ASUS\Downloads\Stop-Sign.v3i.coco\valid\train\images"
train_annotation_dir = r"C:\Users\ASUS\Downloads\Stop-Sign.v3i.coco\valid\train\labels"
valid_image_dir = r"C:\Users\ASUS\Downloads\Stop-Sign.v3i.coco\valid\valid\images"
valid_annotation_dir = r"C:\Users\ASUS\Downloads\Stop-Sign.v3i.coco\valid\valid\labels"

# Create directories for the training and validation sets
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_annotation_dir, exist_ok=True)
os.makedirs(valid_image_dir, exist_ok=True)
os.makedirs(valid_annotation_dir, exist_ok=True)

# List all image files in the original image directory
image_files = [f for f in os.listdir(original_image_dir) if f.endswith('.jpg')]

# Define the split ratio (e.g., 80% for training, 20% for validation)
train_ratio = 0.8
num_images = len(image_files)
num_train = int(train_ratio * num_images)

# Randomly shuffle the list of image files
random.shuffle(image_files)

# Split the images into training and validation sets
train_images = image_files[:num_train]
valid_images = image_files[num_train:]

# Copy the images and corresponding annotation files to their respective directories
for image in train_images:
    annotation = image.replace('.jpg', '.txt')
    shutil.copy(os.path.join(original_image_dir, image), os.path.join(train_image_dir, image))
    shutil.copy(os.path.join(original_annotation_dir, annotation), os.path.join(train_annotation_dir, annotation))

for image in valid_images:
    annotation = image.replace('.jpg', '.txt')
    shutil.copy(os.path.join(original_image_dir, image), os.path.join(valid_image_dir, image))
    shutil.copy(os.path.join(original_annotation_dir, annotation), os.path.join(valid_annotation_dir, annotation))


# In[44]:


import subprocess


# In[78]:


yolov5_command = "python detect.py --source C:/Users/ASUS/Downloads/Stop-Sign.v3i.coco/valid/train/images/download-7-_jpg.rf.501868d99e5127ab254fca619e745a78.jpg --weights C:/Users/ASUS/Downloads/labelImg-master/Labeling/yolov5/runs/train/exp5/weights/best.pt --conf 0.4 --img-size 143 --save-txt"


# In[79]:


result = subprocess.run(yolov5_command, shell=True, capture_output=True, text=True)


# In[80]:


print("Exit Code:", result.returncode)
print("Standard Output:", result.stdout)
print("Standard Error:", result.stderr)


# In[81]:


from IPython.display import Image, display

# Provide the path to the image you want to display
image_path = r"C:\Users\ASUS\Downloads\yolov5-master\yolov5-master\runs\detect\exp7\download-7-_jpg.rf.501868d99e5127ab254fca619e745a78.jpg"

# Display the image in the Jupyter Notebook
display(Image(filename=image_path))

