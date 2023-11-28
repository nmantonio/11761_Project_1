import numpy as np
import cv2
from natsort import natsorted
import os

def show(img): 
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

save_path = r'./res'
images_path = r'./images'
images_list = natsorted(list(os.listdir(images_path)))

# print(images_list)

images = []
images_grayscale = []
for image_name in images_list: 
    # Read images and save in a list
    image = cv2.imread(os.path.join(images_path, image_name))
    images.append(image)

    # Save also grayscale images
    image_grayscale = cv2.imread(os.path.join(images_path, image_name), 0)
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

lower_threshold, upper_threshold = 25, 220
ranged_mask = cv2.inRange(substracted, lower_threshold, upper_threshold)
ranged_image = substracted.copy()
ranged_image[ranged_mask==0] = 0
cv2.imshow('Range', ranged_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(save_path, 'ranged.jpg'), ranged_image)