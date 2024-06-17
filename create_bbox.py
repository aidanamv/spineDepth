import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use the Tkinter backend

dir = "/media/aidana/SpineDepth/PoinTr_dataset/segmented_spinedepth_new/"
save_label_dir = "/media/aidana/SpineDepth/YOLO/labels"
save_image_dir = "/media/aidana/SpineDepth/YOLO/imgs"

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
                        # Compute the oriented bounding box
                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)

                        box_normalized = np.zeros((4,2))
                        box_normalized[:,0] = box[:,0]/image_width
                        box_normalized[:,1] = box[:,1]/image_height


                        # # Draw the oriented bounding box
                        # cv2.drawContours(original_image, [box], 0, (0, 255, 0), 2)
                        # # Display the result
                        # plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                        # plt.title('Oriented Bounding Box')
                        # plt.axis('off')
                        # plt.show()
                        # # # # Display the original image with rectangles
                        # plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                        # plt.title('Bounding Boxes on Original Image')
                        # plt.show()
                        file.write(f"{num} {box_normalized[0][0]:.6f} {box_normalized[0][1]:.6f} {box_normalized[1][0]:.6f} {box_normalized[1][1]:.6f} {box_normalized[2][0]:.6f} {box_normalized[2][1]:.6f} {box_normalized[3][0]:.6f} {box_normalized[3][1]:.6f}\n")

                        num+=1
                    file.close()
