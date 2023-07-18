import albumentations as A
import cv2
import numpy as np
import os

# Directory path where your images are stored
images_directory = r"D:\test"

# Get a list of file names in the directory
file_names = os.listdir(images_directory)

# Loop through each file name and read the images
for file_name in file_names:
    # Create the full file path
    file_path = os.path.join(images_directory, file_name)

    # Read the image using OpenCV
    image = cv2.imread(file_path)


    #horizontal flip
    transform_hori_flip= A.HorizontalFlip(p=1.0)
    hori_transformed = transform_hori_flip(image=image)
    hori_transformed_image = hori_transformed['image']

    #blur
    transform_blur=A.Blur(blur_limit=7, always_apply=False, p=1.0)
    blur_transformed = transform_blur(image=image)
    blur_transformed_image = blur_transformed['image']

    #vertical_flip
    transform_vert_flip=A.VerticalFlip(p=1.0)
    vert_transformed = transform_vert_flip(image=image)
    vert_transformed_image = vert_transformed['image']

    #rotation
    transform_rotate=A.augmentations.geometric.rotate.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, rotate_method='largest_box', crop_border=False, always_apply=False, p=1.0)
    rotate_transformed = transform_rotate(image=image)
    rotate_transformed_image = rotate_transformed['image']

    #random_brightness
    transform_rand_bright=A.RandomBrightness (limit=0.2, p=1.0)
    ranbright_transformed = transform_rand_bright(image=image)
    ranbright_transformed_image = ranbright_transformed['image']

    #random_contrast
    transform_rand_contra=A.RandomContrast(limit=0.2, p=1.0)
    rancont_transformed = transform_rand_contra(image=image)
    rancont_transformed_image = rancont_transformed['image']

    #shearing
    transform_shear=A.Affine(shear={"x":(-30,30),"y":(-30,30)}, p=1.0)
    shear_transformed = transform_shear(image=image)
    shear_transformed_image = shear_transformed['image']

    #color_jitter
    transform_colorjit=A.ColorJitter(p=1.0)
    colorjit_transformed = transform_colorjit(image=image)
    colorjit_transformed_image = colorjit_transformed['image']

    #Gauss_noise
    transform_gauss_noise= A.GaussNoise(p=1.0)
    gauss_noise_transformed = transform_gauss_noise(image=image)
    gauss_noise_transformed_image =  gauss_noise_transformed['image']


    #arrays
    hori_flip_image=np.array(hori_transformed_image, dtype=np.uint8)
    blur_image=np.array(blur_transformed_image, dtype=np.uint8)
    vert_flip_image=np.array(vert_transformed_image, dtype=np.uint8)
    rotate_image=np.array(rotate_transformed_image, dtype=np.uint8)
    rand_bright_image=np.array(ranbright_transformed_image, dtype=np.uint8)
    rand_contra_image=np.array(rancont_transformed_image, dtype=np.uint8)
    shear_image=np.array(shear_transformed_image, dtype=np.uint8)
    colorjit_image=np.array(colorjit_transformed_image, dtype=np.uint8)
    gausnoise_image=np.array(gauss_noise_transformed_image, dtype=np.uint8)

    #writing images
    cv2.imwrite(r"D:/image/horizontal_flip/" + file_name , hori_flip_image)
    cv2.imwrite(r"D:/image/blur/" + file_name , blur_image)
    cv2.imwrite(r"D:/image/vertical_flip/" + file_name , vert_flip_image)
    cv2.imwrite(r"D:/imag/rotation/" + file_name , rotate_image)
    cv2.imwrite(r"D:/image/random_brightness/" + file_name , rand_bright_image)
    cv2.imwrite(r"D:/image/random_contrast/" + file_name , rand_contra_image)
    cv2.imwrite(r"D:/image/sheared/" + file_name , shear_image)
    cv2.imwrite(r"D:/image/color_jitter/" + file_name, colorjit_image)
    cv2.imwrite(r"D:/image/gauss_noise/" + file_name, gausnoise_image)
    
#image=cv2.imread(r"C:\Users\noman\yolov5\Traindata\images\train\GJ17.jpg")
