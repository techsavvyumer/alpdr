import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import *
from tensorflow.keras.models import load_model
from util import get_car, read_license_plate, write_csv
import statistics as stats

# Set the GPU device
# import torch
# torch.cuda.set_device(0)  # Set to your desired GPU number

# Load YOLO models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# Load the character recognition model
model = load_model('updated_tfkmodel.keras')

# Initialize SORT tracker
mot_tracker = Sort()

def process_image(image):
    # detect vehicles
    detections = coco_model(image)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        # Considering only certain classes as vehicles
        vehicles = [2, 3, 5, 7]
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # detect license plates
    license_plates = license_plate_detector(image)[0]
    license_plate_results = []
    lp_crop = None
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        
        # crop license plate from the original image
        license_plate_crop_original = image[int(y1):int(y2), int(x1):int(x2), :]
        lp_crop = license_plate_crop_original
        # assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
        
        if car_id != -1:
            license_plate_results.append((car_id, license_plate_crop_original))
        else:
            continue

    return license_plate_results, lp_crop


# Match contours to license plate or character template
def find_contours(dimensions, img):
    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # Check largest 5 or 15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    ii = cv2.imread('contour.jpg')

    x_cntr_list = []
    img_res = []
    widths = []
    heights = []
    contours = []  # Store contour coordinates here

    # Calculate the middle line of the license plate
    middle_line = img.shape[0] // 2

    # Sort contours based on x-coordinate and then on y-coordinate
    sorted_cntrs = sorted(cntrs, key=lambda c: (c[0][0][0], c[0][0][1]))

    # Separate contours above and below the middle line
    above_middle = []
    below_middle = []
    for cntr in sorted_cntrs:
        x, y, w, h = cv2.boundingRect(cntr)

        if y < middle_line:
            above_middle.append((x, y, w, h, cntr))
        else:
            below_middle.append((x, y, w, h, cntr))

    # Sort contours from left to right
    above_middle = sorted(above_middle, key=lambda c: c[0])
    below_middle = sorted(below_middle, key=lambda c: c[0])

    # Process contours above the middle line
    for x, y, w, h, cntr in above_middle:
        if w > lower_width and w < upper_width and h > lower_height and h < upper_height:
            char = img[y:y+h, x:x+w]
            white_pixels = np.sum(char == 255)
            total_pixels = char.size
            white_percentage = (white_pixels / total_pixels) * 100

            if white_percentage >= 25:
                x_cntr_list.append((x, y))  # stores the (x, y) coordinates of the character's contour
                widths.append(w)
                heights.append(h)
                contours.append((x, y, x+w, y+h))  # Append contour coordinates
                
                char_copy = np.zeros((44, 24))
                # Extracting each character using the enclosing rectangle's coordinates.
                char = cv2.resize(char, (20, 40))

                cv2.rectangle(ii, (x, y), (x+w, y+h), (50, 21, 200), 2)
                plt.imshow(ii, cmap='gray')

                # Make result formatted for classification: invert colors
                char = cv2.subtract(255, char)

                # Resize the image to 24x44 with a black border
                char_copy[2:42, 2:22] = char
                char_copy[0:2, :] = 0
                char_copy[:, 0:2] = 0
                char_copy[42:44, :] = 0
                char_copy[:, 22:24] = 0

                img_res.append(char_copy)  # List that stores the character's binary image (unsorted)

    # Process contours below the middle line
    for x, y, w, h, cntr in below_middle:
        if w > lower_width and w < upper_width and h > lower_height and h < upper_height:
            char = img[y:y+h, x:x+w]
            white_pixels = np.sum(char == 255)
            total_pixels = char.size
            white_percentage = (white_pixels / total_pixels) * 100

            if white_percentage >= 25:
                x_cntr_list.append((x, y))  # stores the (x, y) coordinates of the character's contour
                widths.append(w)
                heights.append(h)
                contours.append((x, y, x+w, y+h))  # Append contour coordinates
                
                char_copy = np.zeros((44, 24))
                # Extracting each character using the enclosing rectangle's coordinates.
                char = cv2.resize(char, (20, 40))

                cv2.rectangle(ii, (x, y), (x+w, y+h), (50, 21, 200), 2)
                plt.imshow(ii, cmap='gray')

                # Make result formatted for classification: invert colors
                char = cv2.subtract(255, char)

                # Resize the image to 24x44 with a black border
                char_copy[2:42, 2:22] = char
                char_copy[0:2, :] = 0
                char_copy[:, 0:2] = 0
                char_copy[42:44, :] = 0
                char_copy[:, 22:24] = 0

                img_res.append(char_copy)  # List that stores the character's binary image (unsorted)
    
    try:
        # Calculate the mode of the widths and heights
        mode_width = stats.mode(widths)
        mode_height = stats.mode(heights)
        print(f"Mode width: {mode_width}, Mode height: {mode_height}")
    
        # Calculate median width and height
        median_width = stats.median(widths)
        median_height = stats.median(heights)
        print(f"Median width: {median_width}, Median height: {median_height}")
    
        # Calculate the mean of the widths and heights
        mean_width = stats.mean(widths)
        mean_height = stats.mean(heights)
        # mean_width = (mode_width + median_width) / 2
        # mean_height = (mode_height + median_height) / 2
        print(f"Mean width: {mean_width}, Mean height: {mean_height}")    
    
        print(f"No# of characters: {len(img_res)}")
    except:
        print("The image quality is too low to detect characters.")
    
    # Filter characters based on width and height deviation from the median
    filtered_img_res = []
    filtered_contours = []
    for char, contour in zip(img_res, contours):
        x1, y1, x2, y2 = contour
        if ((x2 - x1) >= 0.70 * median_width) and ((x2 - x1) <= 1.3 * median_width) and ((y2 - y1) >= 0.70 * median_height) and ((y2 - y1) <= 1.3 * median_height):
            filtered_img_res.append(char)
            filtered_contours.append(contour)

    # Remove contours with a distance of more than 15 pixels between them
    remaining_contours = []
    remaining_filtered_img_res = []
    if len(filtered_contours) > 1:
        center_x = img.shape[1] / 3
        distance = 0
        count = 1
        for i in range(1, len(filtered_contours)):
            if count >= len(filtered_contours):
                break
            x1_prev, _, x2_prev, _ = filtered_contours[count - 1]
            x1_curr, _, x2_curr, _ = filtered_contours[count]
            distance = x1_curr - x2_prev
            print (f"Distance between contours {count} and {count+1}: {distance}")
            if distance <= -175:
                count += 1
                continue
            elif distance <= 15:
                print("if (distance <= 15):")
                remaining_contours.append(filtered_contours[count - 1])
                remaining_filtered_img_res.append(filtered_img_res[count - 1])
                count += 1
            elif x1_curr < center_x:
                print("elif x1_curr < center_x:")
                remaining_contours.append(filtered_contours[count])
                remaining_filtered_img_res.append(filtered_img_res[count])
                count +=2
            elif x1_curr >= (center_x * 2):
                print("elif x1_curr >= center_x:")
                remaining_contours.append(filtered_contours[count - 1])
                remaining_filtered_img_res.append(filtered_img_res[count - 1])
                count +=2
            else:
                print("else:")
                remaining_contours.append(filtered_contours[count - 1])
                remaining_filtered_img_res.append(filtered_img_res[count - 1])
                count += 1

        print("Last contour")
        remaining_contours.append(filtered_contours[-1])
        remaining_filtered_img_res.append(filtered_img_res[-1])

    # plt.show()

    return np.array(remaining_filtered_img_res)#, filtered_contours, remaining_contours


# Segment characters from the license plate
def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # # Make borders white
    # img_binary_lp[0:8,:] = 255
    # img_binary_lp[:,0:25] = 255
    # img_binary_lp[67:75,:] = 255
    # img_binary_lp[:,308:333] = 255
    
    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/10,
                  2*LP_WIDTH/2.5,
                  LP_HEIGHT/20,
                  2*LP_HEIGHT/2.5]
    
    # plt.show()
    cv2.imwrite('contour.jpg', img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)
    # char_list, contours, remaining_contours = find_contours(dimensions, img_binary_lp)
    
    return char_list#, contours, remaining_contours


def fix_dimension(img):
    new_img = np.zeros((28,28,3))
    for i in range(3):
        new_img[:,:,i] = img
    return new_img

def show_results(char_list, model):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char_list): # iterating over the characters
        img_ = cv2.resize(ch, (28,28))
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) # preparing image for the model
        # y_ = model.predict_classes(img)[0] # predicting the class
        predict_x= model.predict(img)
        classes_x = int(np.argmax(predict_x, axis=1)[0])
        character = dic[classes_x] #
        output.append(character) # storing the result in a list
        
    plate_number = ''.join(output)
    
    return plate_number


def main():
    st.title("License Plate Detection and Character Recognition")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Process the image
        lp_results, lp_img_crop = process_image(image)

        # Display results
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if lp_results:
            for _, license_plate_crop in lp_results:
                st.image(license_plate_crop, caption=f"License Plate", use_column_width=True)

                # Segment and recognize characters
                char_list = segment_characters(license_plate_crop)
                lp_number = show_results(char_list, model)

                # for i in range(len(char_list)):
                #     plt.subplot(1, len(char_list), i+1)
                #     plt.imshow(char_list[i], cmap='gray')
                #     plt.axis('off')
                # plt.savefig('segmented_chars.jpg')
                
                if lp_number != "":
                    st.write(f"Recognized License Plate: {lp_number}")
                else:
                    st.warning("The image quality is too low to detect characters.")
        else:
            if lp_img_crop is not None:
                st.image(lp_img_crop, caption=f"License Plate", use_column_width=True)
                char_list = segment_characters(lp_img_crop)
                lp_number = show_results(char_list, model)
            else:
                st.image(image, caption=f"License Plate", use_column_width=True)
                char_list = segment_characters(image)
                lp_number = show_results(char_list, model)
            
            # Segment and recognize characters
            # char_list = segment_characters(lp_img_crop)
            # lp_number = show_results(char_list, model)

            # for i in range(len(char_list)):
            #     plt.subplot(1, len(char_list), i+1)
            #     plt.imshow(char_list[i], cmap='gray')
            #     plt.axis('off')
            # plt.savefig('segmented_chars.jpg')
            
            if lp_number != "":
                st.write(f"Recognized License Plate: {lp_number}")
            else:
                st.warning("The image quality is too low to detect characters.")

if __name__ == "__main__":
    main()
