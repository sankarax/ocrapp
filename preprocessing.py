from scipy.io import savemat
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math

#The functions are explained in the report
def prepro(filepath:str):
    image = cv2.imread(filepath)

    #resizing based on words in the image
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 105,105)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    external_contours = [cnt for i, cnt in enumerate(contours) if hierarchy[0][i][3] == -1]
    bounding_boxes = [cv2.boundingRect(c) for c in external_contours]
    if bounding_boxes:
        max_width = max([box[2] for box in bounding_boxes])
        max_height = max([box[3] for box in bounding_boxes])
        filtered_contours = [c for box, c in zip(bounding_boxes, external_contours) if box[2] > 0.65 * max_width or box[3] > 0.4 * max_height]
    else:
        filtered_contours = []
        
    bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]
    if bounding_boxes:
        x_vals = [x for (x, y, w, h) in bounding_boxes]
        y_vals = [y for (x, y, w, h) in bounding_boxes]
        x_max_vals = [x + w for (x, y, w, h) in bounding_boxes]
        y_max_vals = [y + h for (x, y, w, h) in bounding_boxes]

        x_min = min(x_vals)
        y_min = min(y_vals)
        x_max = max(x_max_vals)
        y_max = max(y_max_vals)

        overall_bounding_box = (x_min, y_min, x_max - x_min, y_max - y_min) 
    else:
        overall_bounding_box = []
    
    y1 = overall_bounding_box[1] - round(0.15*overall_bounding_box[3])
    y2 = overall_bounding_box[1] + round(1.3*overall_bounding_box[3])
    x1 = overall_bounding_box[0] - round(0.1*overall_bounding_box[2])
    x2 = overall_bounding_box[0] + round(1.2*overall_bounding_box[2])
    if x1<=0:
        x1 = 0
    if y1<=0:
        y1=0
    image = image[y1:y2,x1:x2]
    
    #letters contours extraction
    height, width = image.shape[:2]
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img,(2*(height//50)+1, 2*(height//50)+1), 0)
    

    inverted = cv2.bitwise_not(blurred_img)
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(width//height,2))
    enhanced_inverted = clahe.apply(inverted)
    enhanced = cv2.bitwise_not(enhanced_inverted)


    edges = cv2.Canny(enhanced, 50,50)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (round(height/(40)), round(height/(21))))
    dilated_img = cv2.dilate(edges, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    external_contours = [cnt for i, cnt in enumerate(contours) if hierarchy[0][i][3] == -1]
    bounding_boxes = [cv2.boundingRect(c) for c in external_contours]
    filtered_contours = [c for box,c in zip(bounding_boxes, external_contours) if box[2] > height/(3) or box[3] > height/(6)]
    bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]

    #define threshold
    word_gap_threshold = 0.23*height  #adjust based on image resolution and font size

    bounding_boxes_sorted, contours_sorted = map(list, zip(*sorted(zip(bounding_boxes, filtered_contours), key=lambda b: b[0][0])))
    words = []
    current_word = [contours_sorted[0]]
    for i in range(1, len(contours_sorted)):
        prev_box = bounding_boxes_sorted[i - 1]
        curr_box = bounding_boxes_sorted[i]
        #gap bw contours
        gap = curr_box[0] - (prev_box[0]+prev_box[2])
        
        if gap > word_gap_threshold:
            #start new word if gap is large
            words.append(current_word)
            current_word = [contours_sorted[i]]
        else:
            #same word
            current_word.append(contours_sorted[i])
    words.append(current_word)

    output_dir = "preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    padding = 1
    word_idx = 1
    for j, arr in enumerate(words):
        letter_idx = 1
        for i, letter in enumerate(arr):
            x, y, w, h = cv2.boundingRect(letter)
            sidex = w + 2 * padding
            sidey = h + 2 * padding
            if(sidey>sidex):
                extra_direction = True
            else:
                extra_direction = False

            extra_side = abs(sidey-sidex)
            
            x_center = x + w // 2
            y_center = y + h // 2
            x_padded = max(x_center - sidex // 2, 0)
            y_padded = max(y_center - sidey // 2, 0)
            x_padded = min(x_padded, image.shape[1] - sidex)
            y_padded = min(y_padded, image.shape[0] - sidey)

            

            cropped_img = image[y_padded:y_padded + sidey, x_padded:x_padded + sidex]
            #print(f"{word_idx}_{letter_idx}")
            #print(cropped_img.shape)
            #print(extra_side)
            #add extra padding on the sides
            intensity = cropped_img.sum(axis=2)
            max_idx = np.unravel_index(np.argmax(intensity, axis=None), intensity.shape)
            pixel = cropped_img[max_idx[0], max_idx[1], :]  
            padding_color = tuple(int(c) for c in pixel)
            padded_img = cv2.copyMakeBorder(
            cropped_img,
            top=10+int(not(extra_direction))*math.floor(extra_side/2),
            bottom=10+int(not(extra_direction))*math.ceil(extra_side/2),
            left=10+int(extra_direction)*math.floor(extra_side/2),
            right=10+int(extra_direction)*math.ceil(extra_side/2),
            borderType=cv2.BORDER_CONSTANT,
            value=padding_color
            )
            #print(padded_img.shape)
            output_loc = os.path.join(output_dir, f"text{word_idx}_{letter_idx}.png")
            cv2.imwrite(output_loc, padded_img)
            letter_idx = letter_idx + 1
        word_idx = word_idx + 1

    #output_loc = os.path.join(output_dir, f"test.png")
    #cv2.imwrite(output_loc, dilated_img)  #testing
    return edges

if __name__ == "__main__":  
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    final_img = prepro(
        'images\\helloworld.jpeg'
        )

