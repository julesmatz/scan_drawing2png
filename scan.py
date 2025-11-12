# opencv script to scan hand drawn figures (binary black-white color)
# author : jules matz

import cv2
import numpy as np

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

cap = cv2.VideoCapture(0) # try 1, 2, etc. if external webcam

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

    img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                   cv2.THRESH_BINARY_INV, 201, 15)
    binary_image_3ch = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(img_bin, cv2.RETR_CCOMP,
                                   cv2.CHAIN_APPROX_NONE)
    
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 15]
    
    # Iterate over each contour
    for contour in filtered_contours:
        # Get the dominant color of the contour using HSV
        dominant_color = get_dominant_color_hsv(img, contour)

        if dominant_color in COLOR_RGB:
            # Assign the default RGB code for the dominant color
            rgb_code = COLOR_RGB[dominant_color]
            bgr_code = (rgb_code[2], rgb_code[1], rgb_code[0])

            # Draw the contour with the assigned color (filled)
            cv2.drawContours(img, [contour], -1, bgr_code, thickness=cv2.FILLED)

    mask = np.zeros_like(img_bin, dtype=np.uint8)
    cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)


    # Create a white background of the same size as the frame
    white_bg = np.ones_like(img) * 255
    inverted_mask = cv2.bitwise_not(mask)
    white_bg = cv2.bitwise_and(white_bg, white_bg, mask=inverted_mask)
    # Combine the masked image and the white background
    result = cv2.add(white_bg, img)

    cv2.imshow('raw',frame)
    cv2.imshow('filt',result)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    elif k == ord("s"):
        nb_shot = nb_shot+1
        # Create an alpha channel: 255 (opaque) where mask is white, 0 (transparent) where mask is black
        alpha = mask
        # Merge the BGR image with the alpha channel
        result_with_alpha = cv2.merge((img[:,:,0], img[:,:,1], img[:,:,2], alpha))
        cv2.imwrite("figure"+str(nb_shot)+".png", result_with_alpha)

    
cv2.destroyAllWindows()
