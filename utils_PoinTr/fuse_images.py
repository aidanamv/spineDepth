from PIL import Image
import numpy as np
import os


def color_white_pixels(images, colors):
    # Convert image to RGBA mode (with alpha channel)
    white_masks = []
    for image in images:
        image = image.convert("RGBA")
        data = np.array(image)  # Convert image to numpy array

        # Create a mask for white pixels
        white_mask = (data[:, :, 0] == 255) & (data[:, :, 1] == 255) & (data[:, :, 2] == 255)

        white_masks.append(white_mask)

    for white_mask, color in zip(white_masks, colors):
        data[white_mask] = color + (255,)


    # Convert the numpy array back to an image
    colored_image = Image.fromarray(data, mode="RGBA")

    return colored_image


# Open and color white pixels in each image
image_paths = ["mask1.png", "mask2.png", "mask3.png", "mask4.png", "mask5.png"]
dir = "/Volumes/Extreme SSD/PoinTr dataset/segmented_spinedepth_new/Specimen_2/recording_0/cam_1/frame_0"

colors =[(245, 245, 220),  (64, 224, 208),(127, 255, 212), (210, 105, 30),(101, 67, 33)]
images = [Image.open(os.path.join(dir,path)) for path in image_paths]

fused_image = color_white_pixels(images, colors)

data = np.array(fused_image)  # Convert image to numpy array

# Create a mask for white pixels
black_mask = (data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0)
data[black_mask] = (255,255,255) + (255,)

    # Convert the numpy array back to an image
colored_image = Image.fromarray(data, mode="RGBA")

# Save or display the fused image
colored_image.save("fused_image.png")
colored_image.show()