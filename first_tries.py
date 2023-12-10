import numpy as np
import cv2
from natsort import natsorted
import os
from matplotlib import pyplot as plt

save_path = r'./res'
images_path = r'./images'
images_list = natsorted(list(os.listdir(images_path)))

bg_points = np.array([
    (0, 589),
    (27, 529),
    (97, 530),
    (95, 442),
    (137, 446),
    (223, 440),
    (714, 417),
    (854, 402),
    (1179, 411),
    (1288, 412),
    (1384, 418),
    (1480, 418),
    (1566, 409),
    (1920, 398),
    (1920, 0),
    (0, 0)
])

background_mask = np.zeros((1080, 1920), dtype=np.uint8)
cv2.fillPoly(background_mask, [bg_points], color=255)
background_mask = cv2.bitwise_not(background_mask)
cv2.imwrite(os.path.join(save_path, 'background.jpg'), background_mask)

# print(images_list)

images = []
images_grayscale = []
for image_name in images_list: 
    # Read images and save in a list
    image = cv2.imread(os.path.join(images_path, image_name))
    images.append(image)

    # Save also grayscale images
    image_grayscale = cv2.imread(os.path.join(images_path, image_name), 0)
    image_grayscale = image_grayscale
    images_grayscale.append(image_grayscale)    

empty_images = np.array(images_grayscale[8:11])

# Apply GaussianBlur to empty images
empty_images = [cv2.GaussianBlur(image, (5, 5), 0) for image in empty_images]

# Calculate the average image
average_image = np.mean(empty_images, axis=0).astype(np.uint8)

# Display or save the average image as needed
# cv2.imshow('Average Image', average_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite(os.path.join(save_path, 'average_image.jpg'), average_image)

# Example of substraction
substracted = images_grayscale[0] - average_image
# cv2.imshow('Substraction', substracted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite(os.path.join(save_path, 'substracted.jpg'), substracted)

# Setting to 0 values outside [lower_threshold, upper_threshold], trying to ommit extremes
# lower_threshold, upper_threshold = 25, 220
# ranged_mask = cv2.inRange(substracted, lower_threshold, upper_threshold)
# ranged_image = substracted.copy()
# ranged_image[ranged_mask==0] = 0
# cv2.imshow('Range', ranged_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite(os.path.join(save_path, 'ranged.jpg'),ranged_image)

# Trying to use background image
wo_bg = cv2.bitwise_and(substracted, substracted, mask=background_mask)
cv2.imwrite(os.path.join(save_path, 'wo_bg.jpg'),wo_bg)

# # Applying closing
# closing = cv2.morphologyEx(wo_bg, cv2.MORPH_CLOSE, np.ones((1, 1), dtype=np.uint8))
# cv2.imwrite(os.path.join(save_path, 'closing.jpg'),closing)

# Histogram equalization
equ_hist = cv2.equalizeHist(wo_bg)
cv2.imwrite(os.path.join(save_path, 'equalized_hist.jpg'), equ_hist)

# CLAHE 
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
clahed = clahe.apply(wo_bg)
cv2.imwrite(os.path.join(save_path, 'clahed.jpg'), clahed)


# # Otsu thresholding
# blurred = cv2.GaussianBlur(closing, (5, 5), 0)
# _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imwrite(os.path.join(save_path, 'otsu.jpg'),otsu)




