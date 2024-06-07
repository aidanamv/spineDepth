import os
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('TkAgg')  # Use the Tkinter backend
def show_mask(mask, ax, random_color=False):

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

path = '/media/aidana/SpineDepth/YOLO'
save_dir = '/media/aidana/SpineDepth/YOLO/binary_seg'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
bboxes = os.listdir(os.path.join(path, "predictions"))
print(len(bboxes))
for filename in bboxes:
    print(filename)
    image = cv2.imread(os.path.join(path,"images",filename.replace(".txt",".png")))
    bbox = np.loadtxt(os.path.join(path,"predictions",filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    if len(bbox)==5:
        input_point = np.array([[width*bbox[1],height*bbox[2]]])

    else:
        input_point = np.array([[width*bbox[0,1],height*bbox[0,2]]])



    sam_checkpoint = "./segment-anything/checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    input_label = np.array([1])

    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('on')
    # plt.show()

    masks, scores, logit = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    ind = np.argmax(scores)
    print(scores[ind])



        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # show_mask(mask, plt.gca())
        # show_points(input_point, input_label, plt.gca())
        # plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        # plt.axis('off')
        # plt.show()
    binary_image = np.uint8(masks[ind]) * 255
    # Save the binary image
    cv2.imwrite(os.path.join(save_dir,filename.replace(".txt",".png")), binary_image)


