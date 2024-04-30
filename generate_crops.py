import os
import cv2
import numpy as np
from PIL import Image


# Generate and save crops

def create_zoom_out_crops(image_path, base_box, save_directory, zoom_factor=3):
    # base_box is xywh
    # zoom factor is how much you zoom out each time

    os.makedirs(save_directory, exist_ok=True)

    with Image.open(image_path).convert("RGB") as pil_image:
        image = np.array(pil_image)

    # ensure one dimension is not too much longer than the other
    max_aspect_ratio = 3
    if base_box[2] / base_box[3] > max_aspect_ratio:
        base_box = expand_to_aspect_ratio(base_box, max_aspect_ratio)
        # print("making taller")
    elif base_box[2] / base_box[3] < 1/max_aspect_ratio:
        base_box = expand_to_aspect_ratio(base_box, 1/max_aspect_ratio)
        # print("making wider")
    base_box = clamp_box_to_image(base_box, image)

    # zoom out ----------------------------

    # when zooming out, do so on a square box containing the mask bbox
    # this provides a more standard zoom out in the case the bbox is a weird shape
    square_box = expand_to_aspect_ratio(base_box, 1)

    for z in [0, 1, 2]:
        # get crop box, clamped to image
        box = scale_and_clamp_box(square_box, image, zoom_factor**z)
        overlay_box = None if z == 0 else base_box
        
        # take crop and save to file
        image_basename = os.path.basename(image_path)
        save_path = os.path.join(save_directory, f"{image_basename}_zoom{z}.png")
        save_crop(box, image, save_path, overlay_box)

        # stop zooming out if we're nearing the image dimensions
        # if the max dimension is almost the whole image, we hit image size
        # if the min dimension is an appreciable fraction, the next zoom won't do much
        x_frac = box[2] / image.shape[1]
        y_frac = box[3] / image.shape[0]
        if min(x_frac, y_frac) > 0.5 or max(x_frac, y_frac) > 0.8:
            break


def save_crop(box, image, save_path, overlay_box=None, padding_frac=0):
    # box and overlay_box is array formatted XYWH
    # box is allowed to be outside the image bounds, this function will clamp

    x_pad = int(np.round(box[2] * padding_frac))
    y_pad = int(np.round(box[3] * padding_frac))

    x1 = max(0, box[0] - x_pad)
    x2 = min(image.shape[1], box[0] + box[2] + 2*x_pad)
    y1 = max(0, box[1] - y_pad)
    y2 = min(image.shape[0], box[1] + box[3] + 2*y_pad)
    crop = image[y1:y2, x1:x2]

    # convert to bgr for use with cv2
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

    # add overlay
    if overlay_box is not None:
        x, y, w, h = overlay_box
        # translate overlay coords to be relative to this crop
        x = x - x1
        y = y - y1
        line_width = max(1, int(0.002 * crop.shape[0]))
        cv2.rectangle(crop, (x, y), (x+w, y+h), (0,0,255), thickness=line_width)
    
    cv2.imwrite(save_path, crop)



# Utility functions used above
    
    
def clamp_box_to_image(box, image):
    # box is XYWH
    x1 = max(0, box[0])
    y1 = max(0, box[1])
    x2 = min(image.shape[1], box[0] + box[2])
    y2 = min(image.shape[0], box[1] + box[3])
    return [x1, y1, x2-x1, y2-y1]


def expand_to_aspect_ratio(box, aspect_ratio):
    # box is XYWH, aspect ratio is width/height

    current_ratio = box[2] / box[3]

    if current_ratio <= aspect_ratio:
        # need to make wider
        new_width = aspect_ratio * box[3]
        width_to_add = new_width - box[2]
        return [box[0] - 0.5*width_to_add, box[1], box[2] + width_to_add, box[3]]
    else:
        # need to make taller
        new_height = box[2] / aspect_ratio
        height_to_add = new_height - box[3]
        return [box[0], box[1] - 0.5*height_to_add, box[2], box[3] + height_to_add]


def scale_and_clamp_box(box, image, scale):
    # image is the image to clamp the box inside
    center_x = box[0] + 0.5*box[2]
    center_y = box[1] + 0.5*box[3]
    x1 = np.round(scale*(box[0] - center_x) + center_x)
    x2 = np.round(scale*((box[0]+box[2]) - center_x) + center_x)
    y1 = np.round(scale*(box[1] - center_y) + center_y)
    y2 = np.round(scale*((box[1]+box[3]) - center_y) + center_y)

    unclamped = np.array([x1, y1, x2-x1, y2-y1]).astype(int)
    return clamp_box_to_image(unclamped, image)