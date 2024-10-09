
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from skimage import io
from skimage.transform import resize, rotate
from concurrent.futures import ThreadPoolExecutor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Prevent PIL from truncating images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_directory(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Directory {directory} created.")
    else:
        logging.info(f"Directory {directory} already exists.")

def crop_and_resize_image(img, cropx, cropy, img_size=256):
    """Crop and resize an individual image."""
    y, x, channel = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    cropped_img = img[starty:starty + cropy, startx:startx + cropx]
    resized_img = resize(cropped_img, (img_size, img_size))
    return resized_img

def process_images_in_directory(path, new_path, cropx, cropy, img_size=256):
    """Process and save all images in a given directory by cropping and resizing."""
    create_directory(new_path)
    dirs = [l for l in os.listdir(path) if l != '.DS_Store']
    
    with ThreadPoolExecutor() as executor:
        for total, item in enumerate(dirs, start=1):
            try:
                img = io.imread(os.path.join(path, item))
                processed_img = crop_and_resize_image(img, cropx, cropy, img_size)
                io.imsave(os.path.join(new_path, item), processed_img)
                logging.info(f"Processed and saved {item} ({total}/{len(dirs)})")
            except Exception as e:
                logging.error(f"Failed to process image {item}: {e}")

def find_black_images(file_path, df):
    """Check for completely black images in the dataset."""
    lst_imgs = [l for l in df['image']]
    return [1 if np.mean(np.array(Image.open(os.path.join(file_path, img)))) == 0 else 0 for img in lst_imgs]

def rotate_image(img, angle):
    """Rotate an image by a given angle."""
    return rotate(img, angle)

def mirror_image(img):
    """Mirror an image horizontally."""
    return np.fliplr(img)

def augment_images(file_path, lst_imgs, angles, mirror=False):
    """Apply rotations and mirroring to a list of images for augmentation."""
    for img_name in lst_imgs:
        try:
            img = io.imread(os.path.join(file_path, img_name))
            
            if mirror:
                mirrored_img = mirror_image(img)
                io.imsave(os.path.join(file_path, f"mirrored_{img_name}"), mirrored_img)
                logging.info(f"Mirrored and saved {img_name}")

            for angle in angles:
                rotated_img = rotate_image(img, angle)
                io.imsave(os.path.join(file_path, f"rotated_{angle}_{img_name}"), rotated_img)
                logging.info(f"Rotated {angle} degrees and saved {img_name}")
                
        except Exception as e:
            logging.error(f"Error augmenting image {img_name}: {e}")

def convert_images_to_arrays(file_path, df, column):
    """Convert images to arrays for model training."""
    lst_imgs = [l for l in df[column]]
    return np.array([np.array(Image.open(os.path.join(file_path, img))) for img in lst_imgs])

def save_to_array(file_name, array):
    """Save an array to a .npy file."""
    np.save(file_name, array)
    logging.info(f"Saved array to {file_name}")

if __name__ == '__main__':
    # Example usage of preprocessing functions

    # Crop and resize training and test images
    process_images_in_directory(path='../data/train/', new_path='../data/train-resized-256/', cropx=1800, cropy=1800, img_size=256)
    process_images_in_directory(path='../data/test/', new_path='../data/test-resized-256/', cropx=1800, cropy=1800, img_size=256)

    # Identify black images in the training set
    train_labels = pd.read_csv('../labels/trainLabels.csv')
    train_labels['image'] = train_labels['image'] + '.jpeg'
    train_labels['black'] = find_black_images('../data/train-resized-256/', train_labels)
    train_labels = train_labels.loc[train_labels['black'] == 0]
    train_labels.to_csv('trainLabels_filtered.csv', index=False)
    logging.info("Filtered out black images and saved to trainLabels_filtered.csv")

    # Augment data with rotations and mirroring
    logging.info("Starting data augmentation")
    augment_images(file_path='../data/train-resized-256/', lst_imgs=train_labels['image'], angles=[90, 180, 270], mirror=True)

    # Convert images to arrays and save
    logging.info("Converting images to arrays")
    X_train = convert_images_to_arrays(file_path='../data/train-resized-256/', df=train_labels, column='image')
    save_to_array('../data/X_train.npy', X_train)
    logging.info("Preprocessing completed.")
