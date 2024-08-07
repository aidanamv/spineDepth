import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use the Tkinter backend

dir = "/Volumes/SpineDepth/PoinTr_dataset/segmented_spinedepth_new/"
save_label_dir = "/Volumes/SpineDepth/YOLO/labels"
save_image_dir = "/Volumes/SpineDepth/YOLO/imgs"

if not os.path.exists(save_image_dir):
    os.makedirs(save_image_dir)
if not os.path.exists(save_label_dir):
    os.makedirs(save_label_dir)
specimens = os.listdir(dir)

for specimen in specimens:
    recordings = os.listdir(os.path.join(dir,specimen))
    for recording in recordings:
        cam_nums = os.listdir(os.path.join(dir, specimen, recording))
        for cam_num in cam_nums:
            frames = os.listdir(os.path.join(dir, specimen, recording,cam_num))
            for frame in frames:
                print(specimen, recording, cam_num, frame)
                # Paths to the binary mask and the original image
                original_image_path = os.path.join(dir, specimen, recording,cam_num, frame, "image.png")

                # Load the binary mask
                binary_mask1 = cv2.imread(os.path.join(dir, specimen, recording,cam_num, frame, "mask1.png"), cv2.IMREAD_GRAYSCALE)
                binary_mask2 = cv2.imread(os.path.join(dir, specimen, recording,cam_num, frame, "mask2.png"), cv2.IMREAD_GRAYSCALE)
                binary_mask3 = cv2.imread(os.path.join(dir, specimen, recording,cam_num, frame, "mask3.png"), cv2.IMREAD_GRAYSCALE)
                binary_mask4 = cv2.imread(os.path.join(dir, specimen, recording,cam_num, frame, "mask4.png"), cv2.IMREAD_GRAYSCALE)
                binary_mask5 = cv2.imread(os.path.join(dir, specimen, recording,cam_num, frame, "mask5.png"), cv2.IMREAD_GRAYSCALE)


                # Load the original image
                original_image = cv2.imread(original_image_path)
                cv2.imwrite(os.path.join(save_image_dir,'{}_{}_{}_{}.png'.format(specimen,recording,cam_num, frame)), original_image)

                # Check if the original image is loaded correctly
                if original_image is None:
                    raise FileNotFoundError(f"Unable to load original image at {original_image_path}")

                # Find contours in the binary mask
                contours1, _ = cv2.findContours(binary_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours2, _ = cv2.findContours(binary_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours3, _ = cv2.findContours(binary_mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours4, _ = cv2.findContours(binary_mask4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours5, _ = cv2.findContours(binary_mask5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour1 = max(contours1, key=cv2.contourArea)
                largest_contour2 = max(contours2, key=cv2.contourArea)
                largest_contour3 = max(contours3, key=cv2.contourArea)
                largest_contour4 = max(contours4, key=cv2.contourArea)
                largest_contour5 = max(contours5, key=cv2.contourArea)

                # Image dimensions
                image_height, image_width = original_image.shape[:2]

                contours = [largest_contour1 ,largest_contour2 ,largest_contour3 , largest_contour4,largest_contour5]

                # Draw rectangles around each contour on the original image
                num =0
                with open(os.path.join(save_label_dir, '{}_{}_{}_{}.txt'.format(specimen, recording, cam_num, frame)),
                          'w') as file:

                    for contour in contours:
                        # Get the bounding box coordinates
                        x, y, w, h = cv2.boundingRect(contour)
                        box_normalized = np.array([(x/w, y/h, (x+w)/w, (y+h)/h)])


                        # Draw the bounding box on the binary mask for visualization (optional)
                        bounding_box_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                        cv2.rectangle(bounding_box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)


                        top_left = (x, y)
                        top_right = (x + w, y)
                        bottom_left = (x, y + h)
                        bottom_right = (x + w, y + h)

                        # Normalize the bounding box coordinates
                        top_left_norm = (top_left[0] / h, top_left[1] / w)
                        top_right_norm = (top_right[0] / h, top_right[1] / w)
                        bottom_left_norm = (bottom_left[0] / h, bottom_left[1] / w)
                        bottom_right_norm = (bottom_right[0] / h, bottom_right[1] / w)

                        file.write(
                            f"{num} {top_left_norm[0]} {top_left_norm[1]} {top_right_norm[0]} {top_right_norm[1]} {bottom_left_norm[0]} {bottom_left_norm[1]} {bottom_right_norm[0]} {bottom_right_norm[1]}\n")



                        num+=1
                    file.close()
