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

for specimen in specimens[8:]:
    recordings = os.listdir(os.path.join(dir,specimen))
    for recording in recordings:
        cam_nums = os.listdir(os.path.join(dir, specimen, recording))
        for cam_num in cam_nums:
            frames = os.listdir(os.path.join(dir, specimen, recording,cam_num))
            for frame in frames:
                # Paths to the binary mask and the original image
                binary_mask_path = os.path.join(dir, specimen, recording,cam_num, frame, "mask.png")
                original_image_path = os.path.join(dir, specimen, recording,cam_num, frame, "image.png")

                # Load the binary mask
                binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)

                # Check if the binary mask is loaded correctly
                if binary_mask is None:
                    raise FileNotFoundError(f"Unable to load binary mask at {binary_mask_path}")

                # Load the original image
                original_image = cv2.imread(original_image_path)
                cv2.imwrite(os.path.join(save_image_dir,'{}_{}_{}_{}.png'.format(specimen,recording,cam_num, frame)), original_image)

                # Check if the original image is loaded correctly
                if original_image is None:
                    raise FileNotFoundError(f"Unable to load original image at {original_image_path}")

                # Find contours in the binary mask
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Image dimensions
                image_height, image_width = original_image.shape[:2]

                # Draw rectangles around each contour on the original image
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Normalize the coordinates and dimensions
                    x_norm = x / image_width
                    y_norm = y / image_height
                    w_norm = w / image_width
                    h_norm = h / image_height

                    # Calculate the center of the bounding box
                    center_x = x_norm + w_norm / 2
                    center_y = y_norm + h_norm / 2

                    # Calculate the distances from the center to each side of the bounding box
                    distance_left = center_x - x_norm
                    distance_right = x_norm + w_norm - center_x
                    distance_top = center_y - y_norm
                    distance_bottom = y_norm + h_norm - center_y

                    # Debugging output
                    print(f"Normalized bounding box: x={x_norm}, y={y_norm}, w={w_norm}, h={h_norm}")
                    print(f"Center: ({center_x}, {center_y})")
                    print(
                        f"Distances from center to sides: left={distance_left}, right={distance_right}, top={distance_top}, bottom={distance_bottom}")

                    # Draw the rectangle on the original image
                    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # # Display the original image with rectangles
                # plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                # plt.title('Bounding Boxes on Original Image')
                # plt.show()

                with open(os.path.join(save_label_dir,'{}_{}_{}_{}.txt'.format(specimen,recording,cam_num, frame)), 'w') as file:
                    file.write(f"0 {center_x:.6f} {center_y:.6f} {distance_left:.6f} {distance_top:.6f}\n")

