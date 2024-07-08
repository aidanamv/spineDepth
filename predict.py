from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
matplotlib.use('Qt5Agg')

# Load a model
model = YOLO("best.pt")  # pretrained YOLOv8n model
images_dir = "/Users/aidanamassalimova/Documents/YOLO_dataset/images"
labels_dir = "/Users/aidanamassalimova/Documents/YOLO_dataset/labels"
files = os.listdir(images_dir)
random.shuffle(files)
for file in files:
    print(file)
    file = "Specimen_3_recording_0_cam_0_frame_0.png"
    labels = np.loadtxt(os.path.join(labels_dir, file.replace(".png",".txt")), delimiter=" ")
    results = model(os.path.join(images_dir,file))  # return a list of Results objects
    for result in results:
        boxes = result.boxes
        if all(i >= 0.9 for i in boxes.conf):
            result.show()
            print(result.boxes.conf)
            masks = result.masks.data.cpu().numpy()

            mask1 = masks[0,:,:].reshape(masks.shape[1], masks.shape[2])
            mask2 = masks[1,:,:].reshape(masks.shape[1], masks.shape[2])
            mask3 = masks[2,:,:].reshape(masks.shape[1], masks.shape[2])
            mask4 = masks[3,:,:].reshape(masks.shape[1], masks.shape[2])
            mask5 = masks[4,:,:].reshape(masks.shape[1], masks.shape[2])

            label1 = labels[0,1:9]
            label2 = labels[1,1:9]
            label3 = labels[2,1:9]
            label4 = labels[3,1:9]
            label5 = labels[4,1:9]


            box1_gt = [label1[0] * 1920, label1[1] * 1080, label1[2] * 1920, label1[3]*1080, label1[4]*1920, label1[5]*1080, label1[6]*1920, label1[7]*1080]
            box2_gt = [label2[0] * 1920, label2[1] * 1080, label2[2] * 1920, label2[3]*1080, label2[4]*1920, label2[5]*1080, label2[6]*1920, label2[7]*1080]
            box3_gt = [label3[0] * 1920, label3[1] * 1080, label3[2] * 1920, label3[3]*1080, label3[4]*1920, label3[5]*1080, label3[6]*1920, label3[7]*1080]
            box4_gt = [label4[0] * 1920, label4[1] * 1080, label4[2] * 1920, label4[3]*1080, label4[4]*1920, label4[5]*1080, label4[6]*1920, label4[7]*1080]
            box5_gt = [label5[0] * 1920, label5[1] * 1080, label5[2] * 1920, label5[3]*1080, label5[4]*1920, label5[5]*1080, label5[6]*1920, label5[7]*1080]


            box1 = boxes[0].xyxy
            box2 = boxes[1].xyxy
            box3 = boxes[2].xyxy
            box4 = boxes[3].xyxy
            box5 = boxes[4].xyxy

            box1_np = box1[0].numpy()
            box2_np = box2[0].numpy()
            box3_np = box3[0].numpy()
            box4_np = box4[0].numpy()
            box5_np = box5[0].numpy()

            original_image = cv2.imread(os.path.join(images_dir,file))
            window_name = 'Image'

            mask1 = cv2.resize(mask1, (1920, 1080))
            mask2 = cv2.resize(mask2, (1920, 1080))
            mask3 = cv2.resize(mask3, (1920, 1080))
            mask4 = cv2.resize(mask4, (1920, 1080))
            mask5 = cv2.resize(mask5, (1920, 1080))

            val1, val2,val3,val4,val5 = 0.5, 1, 1.5, 2, 2.5
            mask1[mask1 > 0] = val1
            mask1[mask2 > 0] = val2
            mask1[mask3 > 0] = val3
            mask1[mask4 > 0] = val4
            mask1[mask5 > 0] = val5

          #  image = cv2.rectangle(original_image, pt1=(int(box1_gt[0]),int(box1_gt[1])), pt2=(int(box1_gt[2]),int(box1_gt[3])),color=(0,255,0),thickness=2)
            image = cv2.rectangle(original_image, pt1=(int(box2_gt[0]),int(box2_gt[1])), pt2=(int(box2_gt[2]),int(box2_gt[3])),color=(0,255,0),thickness=2)
            image = cv2.rectangle(image, pt1=(int(box3_gt[0]),int(box3_gt[1])), pt2=(int(box3_gt[2]),int(box3_gt[3])),color=(0,255,0),thickness=2)
            image = cv2.rectangle(image, pt1=(int(box4_gt[0]),int(box4_gt[1])), pt2=(int(box4_gt[2]),int(box4_gt[3])),color=(0,255,0),thickness=2)
            image = cv2.rectangle(image, pt1=(int(box5_gt[0]),int(box5_gt[1])), pt2=(int(box5_gt[2]),int(box5_gt[3])),color=(0,255,0),thickness=2)
            
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            plt.imshow(mask1, cmap='jet', alpha=0.5)  # interpolation='none'


            plt.show()






