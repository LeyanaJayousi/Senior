r"""
Creates segmentation masks.
"""

import numpy as np
import cv2
import shutil
import os
import pandas as pd
import paths
import numpy as np


def create_shape_mask(shape, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)# Initialize a blank mask with the same height and width as the image
    if shape == 'circle':#creating segmentation masks according to the shape of the traffic sign
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        radius = min(abs(x2 - x1), abs(y2 - y1)) // 2
        cv2.circle(mask, center, radius, 255, -1)
    elif shape == 'triangle':
        points = np.array([[x1, y2], [x2, y2], [(x1 + x2) // 2, y1]], np.int32)
        cv2.fillPoly(mask, [points], 255)
    elif shape == 'inverse_triangle':
        points = np.array([[x1, y1], [x2, y1], [(x1 + x2) // 2, y2]], np.int32)
        cv2.fillPoly(mask, [points], 255)
    elif shape == 'octagon':
        points = np.array([[x1 + (x2 - x1) * 0.15, y1],
                       [x2 - (x2 - x1) * 0.15, y1],
                       [x2, y1 + (y2 - y1) * 0.3],
                       [x2, y2 - (y2 - y1) * 0.3],
                       [x2 - (x2 - x1) * 0.15, y2],
                       [x1 + (x2 - x1) * 0.15, y2],
                       [x1, y2 - (y2 - y1) * 0.3],
                       [x1, y1 + (y2 - y1) * 0.3]], np.int32)
        cv2.fillPoly(mask, [points], 255)
    elif shape == 'diamond':
        points = np.array([[(x1 + x2) // 2, y1], [x2, (y1 + y2) // 2],
                           [(x1 + x2) // 2, y2], [x1, (y1 + y2) // 2]], np.int32)
        cv2.fillPoly(mask, [points], 255)
    return mask

def create_and_copy_folders(source_train_path, source_mask_path, destination_path):
    """
    Creates 'Images' and 'Masks' directories in the specified destination path
    and copies image files from the source training and mask paths into these
    directories, renaming them sequentially.
    """
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    train_dir = os.path.join(destination_path, 'Images')
    mask_dir = os.path.join(destination_path, 'Masks')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    train_image_counter = 1
    for subdir, _, files in os.walk(source_train_path):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                shutil.copy(os.path.join(subdir, file), os.path.join(
                    train_dir, f"Image{train_image_counter}.jpg"))
                train_image_counter += 1

    mask_image_counter = 1
    for subdir, _, files in os.walk(source_mask_path):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                shutil.copy(os.path.join(subdir, file), os.path.join(
                    mask_dir, f"Image{mask_image_counter}_label.png"))
                mask_image_counter += 1

                
if __name__ == "__main__":

    
    csv_file = paths.train_csv# Read the CSV file containing image data
    df = pd.read_csv(csv_file)


    output_dir = "segmentation_masks"#Create directories for segmentation masks
    for i in range(20):
        os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)


    for index, row in df.iterrows():
        image_path = row['Path']
        image_path = image_path.replace('/', '\\')
        image_path = os.path.join('data', 'new_dataset', image_path)
        
        if not os.path.exists(image_path):
            #print(f"Image not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            #print(f"Failed to load image: {image_path}")
            continue
        #print(f"Loaded image: {image_path}, shape: {image.shape}")

        x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']# Extract ROI coordinates and ClassId from the CSV


        class_id = row['ClassId']


        if class_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17]: # Determine the shape type based on the class ID
            shape = 'circle'
        elif class_id in [11, 18, 19]:
            shape = 'triangle'
        elif class_id == 13:
            shape = 'inverse_triangle'
        elif class_id == 14:
            shape = 'octagon'
        elif class_id == 12:
            shape = 'diamond'
        else:
            print(f"Invalid class ID: {class_id}")
            continue


        mask = create_shape_mask(shape, image.shape) # Create the shape mask for the image


        mask_filename = os.path.basename(image_path).replace('.jpg', f'_{shape}_mask.jpg')#Generating the file and saving results
        mask_path = os.path.join(output_dir, str(class_id), mask_filename)
        cv2.imwrite(mask_path, mask)
    
    

    # Destination path
    destination_path =  os.path.join("dataDIR")
    source_mask =  os.path.join("segmentation_masks")

    create_and_copy_folders(paths.train_new, source_mask, destination_path)