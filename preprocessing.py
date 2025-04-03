
import cv2
import numpy as np
import os

def prepro(filepath:str):
    image = cv2.imread(filepath)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blurred_img, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_img = cv2.dilate(edges, kernel, iterations=2)
    contours, hier = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_dir = "preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    padding = 10
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 30 and h > 10:
            side = max(w, h) + 2 * padding
            x_center = x + w // 2
            y_center = y + h // 2
            
            x_padded = max(x_center - side // 2, 0)
            y_padded = max(y_center - side // 2, 0)
            x_padded = min(x_padded, image.shape[1] - side)
            y_padded = min(y_padded, image.shape[0] - side)

            cropped_img = image[y_padded:y_padded + side, x_padded:x_padded + side]

            output_loc = os.path.join(output_dir, f"text{i}.jpg")
            cv2.imwrite(output_loc, cropped_img)
    
    return cropped_img

if __name__ == "main":
    final_img = prepro('images/abc.jpeg')

