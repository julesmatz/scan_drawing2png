# OpenCV script to scan hand drawn figures
# author: jules matz

import cv2
import numpy as np

# parameters
CAMERA_NB = 0 # try 1, 2, etc. if external webcam
COLOR = False
DESIRED_WIDTH = 800 # pixels

def get_dominant_color_hsv(image, contour):
    # Create a blank mask
    mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)

    # Draw the contour on the mask, filled with white
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Convert the masked region to HSV
    hsv_masked = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

    # Calculate the average hue, saturation, and value
    avg_hue = np.mean(hsv_masked[..., 0][mask == 255])
    avg_saturation = np.mean(hsv_masked[..., 1][mask == 255])
    avg_value = np.mean(hsv_masked[..., 2][mask == 255])

    # Determine the dominant color based on the average hue
    if avg_hue < 15 or avg_hue > 170:
        return 'red'
    elif 100 <= avg_hue < 140:
        return 'blue'
    elif avg_value < 50:
        return 'black'
    elif avg_saturation < 50:
        return 'gray'
    else:
        return 'green'

cap = cv2.VideoCapture(CAMERA_NB)

# Define default RGB codes for each color
COLOR_RGB = {
    'red': (250, 24, 24),
    'green': (45, 250, 45),
    'blue': (41, 92, 247),
    'black': (0, 0, 0),
    'gray': (128, 128, 128)
}

i=0
nb_shot = 0
while(1):
    # Take each frame
    ret, frame = cap.read()
    img = frame.copy()
    # Convert BGR to grayscale and HSV
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # binarize using adaptative threshold
    img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                   cv2.THRESH_BINARY_INV, 201, 15)
    # smooth by successive erosion and dilation
    kernel = np.ones((3,3),np.uint8)
    for i1 in range(8):
        img_bin = cv2.erode(img_bin,kernel,iterations = 1)
        img_bin = cv2.dilate(img_bin,kernel,iterations = 1)

    binary_image_3ch = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(img_bin, cv2.RETR_CCOMP,
                                   cv2.CHAIN_APPROX_NONE)
    
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 15]
    
    # Iterate over each contour
    xmin = img.shape[1] # width
    ymin = img.shape[0] # height
    xmax = 0
    ymax = 0
    for contour in filtered_contours:
        # get bounds
        min_val = np.min(contour,0)
        max_val = np.max(contour,0)
        xmin = min_val[0][1] if min_val[0][1] < xmin else xmin
        ymin = min_val[0][0] if min_val[0][0] < ymin else ymin
        xmax = max_val[0][1] if max_val[0][1] > xmax else xmax
        ymax = max_val[0][0] if max_val[0][0] > ymax else ymax
        if COLOR:
            # Get the dominant color of the contour using HSV
            dominant_color = get_dominant_color_hsv(img, contour)

            if dominant_color in COLOR_RGB:
                # Assign the default RGB code for the dominant color
                rgb_code = COLOR_RGB[dominant_color]
                bgr_code = (rgb_code[2], rgb_code[1], rgb_code[0])

                # Draw the contour with the assigned color (filled)
                cv2.drawContours(img, [contour], -1, bgr_code, thickness=cv2.FILLED)
        else:
            cv2.drawContours(img, [contour], -1, [0,0,0], thickness=cv2.FILLED)

    mask = np.zeros_like(img_bin, dtype=np.uint8)
    cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)


    # Create a white background of the same size as the frame
    white_bg = np.ones_like(img) * 255
    inverted_mask = cv2.bitwise_not(mask)
    white_bg = cv2.bitwise_and(white_bg, white_bg, mask=inverted_mask)
    # Combine the masked image and the white background
    result = cv2.add(white_bg, img)

    cv2.imshow('input',frame)
    cv2.imshow('output',result)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    elif k == ord("s"):
        nb_shot = nb_shot+1
        # Create an alpha channel: 255 (opaque) where mask is white, 0 (transparent) where mask is black
        alpha = mask
        # Merge the BGR image with the alpha channel
        img_alpha = cv2.merge((img[:,:,0], img[:,:,1], img[:,:,2], alpha))
        # crop
        img_alpha = img_alpha[xmin:xmax, ymin:ymax]
        # scale to DESIRED_WIDTH
        cropped_img_width = img_alpha.shape[1]
        cropped_img_height = img_alpha.shape[0]
        scale_factor = DESIRED_WIDTH / cropped_img_width
        img_alpha = cv2.resize(img_alpha,(DESIRED_WIDTH, int(scale_factor*cropped_img_height)), interpolation = cv2.INTER_CUBIC)
        # save image
        cv2.imwrite("figure"+str(nb_shot)+".png", img_alpha)

    
cv2.destroyAllWindows()
