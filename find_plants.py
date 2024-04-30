"""
This is a one-file implementation of pipeline_v3, for ease of use.
Some changes:
- not using dist_to_side factor in the prominence function, doesn't really seem to matter
- process_image() returns bboxes (for the full res image), visualization is done by functions in generate_crops.py

TO USE IN A JUPYTER NOTEBOOK:

SETUP MODELS ONCE
from find_plants import load_models, process_image
models = load_models()

RUN FOR EACH IMAGE
bboxes = process_image("image/path/here", models, show_steps=True, output_json_file="output.json")

"""


import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import os
import json
import logging

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, CLIPSegProcessor, CLIPSegForImageSegmentation, CLIPProcessor, CLIPModel



def load_models():
    """Loads the ML models we use for the pipeline
    
    Returns: Dictionary, where key is one of "MobileSam", "CLIP", "ClipSeg", "DepthAnything",
    and value is a dictionary with keys "model", "processor" (processor missing for MobileSAM),
    which can be used to run the model. Pass this dictionary to run_pipeline() to use the pipeline.
    """

    models = {}

    # set up segmentation model
    model_type = "vit_t"
    sam_checkpoint = "./weights/mobile_sam.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    models["MobileSAM"] = {"model": mobile_sam}
    print("MobileSAM Loaded")

    # set up clip model
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    models["CLIP"] = {"model": clip_model, "processor": clip_processor}
    print("CLIP loaded")

    # set up clipseg model
    clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    models["ClipSeg"] = {"model": clipseg_model, "processor": clipseg_processor}
    print("ClipSeg loaded")

    # set up depth model
    depth_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
    models["DepthAnything"] = {"model": depth_model, "processor": depth_processor}
    print("DepthAnything Loaded")

    return models




def load_image(path):
    with Image.open(path).convert("RGB") as img:
        full_res_image = np.array(img)

    # resize so not too big for GPU
    max_width = 1024
    height, width = full_res_image.shape[:2]
    new_width = max_width if width > max_width else width
    image_shrink_factor = new_width / width  # will be used later for sampling from full res image
    new_height = int(height * image_shrink_factor)
    image = cv2.resize(full_res_image, (new_width, new_height))

    return image, full_res_image




def process_image(image_path, models, show_steps=True, output_json_file=None):
    """Returns a list of bboxes, with ClipSeg flower results first, and then sorted by prominence.
    Each bbox is a list with 4 numbers: X Y width height.
    Also saves this list of bboxes to a json file, if the json file is specified, appending on to what exists already.
    """

    # load image
    image, full_res_image = load_image(image_path)
    if show_steps:
        show(image, title="Input Image")

    # find flowers using clipseg
    clip_prompts = ["flower", "leaf", "sky", "rock", "dirt", "animal", "person", "human being"]
    flower_bboxes = find_objects(image, models["ClipSeg"]["processor"], models["ClipSeg"]["model"],
                                 clip_prompts, target_prompt="flower", display_results=show_steps,
                                 logit_threshold=0.4)  # this threshold seems to work fine

    # get segmentation masks
    mask_data, unfiltered_mask_data = get_masks(image, models["MobileSAM"]["model"])
    if show_steps:
        show(image, unfiltered_mask_data, title="Unfiltered Masks")
        show(image, mask_data, title="Filtered Masks")

    # get depth
    inverse_depth = get_inverse_depth(image, models["DepthAnything"]["processor"], models["DepthAnything"]["model"])
    if show_steps:
        show(inverse_depth, title="Depth Map")

    # # run segmentation on the depth map?
    # depth_rgb = (np.stack((inverse_depth, inverse_depth, inverse_depth), axis=2) * 255).astype("uint8")
    # depth_mask_data, unfiltered_depth_mask_data = get_masks(depth_rgb)
    # show(image, unfiltered_depth_mask_data, title="Unfiltered Masks from Depth Input")
    # show(image, depth_mask_data, "Filtered Masks from Depth Input")

    # sort masks by prominence
    for mask_object in mask_data:
        mask_object['prominence'] = get_prominence(mask_object, inverse_depth)
    mask_data.sort(key=lambda x: x['prominence'], reverse=True)
    if show_steps:
        show(image, mask_data[:10], title="Top 10 Most Prominent Masks")

    # combine clipseg bboxes with MobileSAM bboxes
    initial_bboxes = flower_bboxes + [mask_object["bbox"] for mask_object in mask_data]

    # BBOX processing -----------------------------------------------
    # convert to full resolution image bboxes
    scale_factor = full_res_image.shape[0] / image.shape[0]
    bboxes = [(np.array(box) * scale_factor).astype(int).tolist() for box in initial_bboxes]
    # filter out duplicate bboxes (ClipSeg flower bboxes might be duplicates of SAM bboxes)
    bboxes = deduplicate_boxes(bboxes, iou_threshold=0.25)
    # filter out super tiny bboxes
    min_bbox_area = 10000
    bboxes = list(filter(lambda x: x[2] * x[3] >= min_bbox_area, bboxes))
    # filter out non plant bboxes
    plant_bboxes = []
    for box in bboxes:
        crop = get_crop(box, full_res_image)
        crop_class = classify(crop, models["CLIP"]["processor"], models["CLIP"]["model"], ["plant", "flower", "fruit", "sky", "dirt", "human", "rock", "water", "vehicle"])
        is_a_plant = crop_class in ["plant", "flower", "fruit"]
        if is_a_plant:
            plant_bboxes.append(box)
        if show_steps:
            show(crop, title=f"{crop_class} - {'keeping' if is_a_plant else 'rejecting'}")

    # save to json file
    if output_json_file:
        existing_json = []
        # check if data already exists in the output file, and if so, add on to it
        if os.path.exists(output_json_file):
            with open(output_json_file) as file:
                existing_json = json.load(file)
        # add this image's results
        existing_json.append({'image_path': image_path, 'bboxes': plant_bboxes})
        # save
        with open(output_json_file, mode='w') as file:
            # don't sort the json, so that the order of interest is preserved
            json.dump(existing_json, file, indent=2)

    return plant_bboxes

    # generate output visualization
    # with PdfPages(pdf_savepath) if pdf_savepath else nullcontext() as pdf:
    #     # pdf will equal None if we aren't saving
    #     if pdf_savepath:
    #         plt.figure(figsize=(9,9))
    #         plt.title("Original Image")
    #         plt.imshow(full_res_image)
    #         plt.axis('off')
    #         plt.savefig(pdf, format='pdf')
    #         plt.close()  # don't show this image in the notebook, only for the PDF
    #     for bbox in bboxes:
    #         plot_crops(bbox, image, full_res_image, zoom_factor=3, pdf_file_object=pdf)




# Useful functions for displaying results
            
def show(image, mask_data=None, title="", figsize=(7,5)):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(image)
    if mask_data is not None:
        show_anns(mask_data)
    plt.axis('off')
    plt.show()
    

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                  sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.6]])
        img[m] = color_mask
    ax.imshow(img)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)




# Run DepthAnything model

def get_inverse_depth(np_image, depth_processor, depth_model):
    # prepare image for the model
    image = Image.fromarray(np_image)
    inputs = depth_processor(images=image, return_tensors="pt")

    # run model
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # normalize and return
    output = prediction.squeeze().cpu().numpy()
    output = output / np.max(output)
    return output






# Flower / object search with ClipSeg

def find_objects(np_image, clipseg_processor, clipseg_model, prompts, target_prompt="flower", display_results=True, logit_threshold=0.25):
    # threshold between 0 and 1, closer to 0 means more areas will be selected
    assert target_prompt in prompts, "target prompt must be one of the prompts"

    # image is a numpy image, convert to PIL
    image = Image.fromarray(np_image)
    
    # prepare clipseg model inputs
    logging.disable(logging.WARNING)  # get rid of annoying padding message
    inputs = clipseg_processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
    logging.disable(logging.NOTSET)

    # predict
    with torch.no_grad():
        outputs = clipseg_model(**inputs)
        preds = outputs.logits.unsqueeze(1)

    # get resized logits heatmap for target prompt and find bounding boxes
    target_prompt_idx = prompts.index(target_prompt)
    target_logits_square = torch.sigmoid(preds[target_prompt_idx][0]).numpy()
    target_logits = cv2.resize(target_logits_square, (image.width, image.height))
    
    bboxes = get_bounding_boxes(target_logits, logit_threshold)

    # display
    if display_results:
        plt.figure(figsize=(7,5))
        plt.axis("off")
        plt.title(f"Found objects for prompt: {target_prompt}")
        plt.imshow(image)
        plt.imshow(target_logits, vmin=0, vmax=1, alpha=0.5)
        for bbox in bboxes:
            x,y,w,h = bbox
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=1))
    
    return bboxes


def get_bounding_boxes(heatmap, logit_threshold=0.4):
    # heatmap should range from 0 to 1
    assert np.min(heatmap) >= 0 and np.max(heatmap) <= 1, "Heatmap values should range from 0 to 1"

    # based on this method: https://stackoverflow.com/questions/58419893/generating-bounding-boxes-from-heatmap-data

    # blur the sigmoid logits for more reliable thresholding
    blur = cv2.GaussianBlur(heatmap, (51,51), 0)

    # threshold
    thresholded = cv2.threshold((255*blur).astype("uint8"), 255*logit_threshold, 255, cv2.THRESH_BINARY)[1]
    # thresholded = cv2.threshold((255*sigmoid_logits).astype("uint8"), 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # get contours / bounding boxes
    bboxes = []
    contours = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for c in contours:
        bboxes.append(cv2.boundingRect(c)) # x,y,w,h
    
    return bboxes




# Mask generation and filtering


def get_masks(np_image, sam_model):
  mask_generator = SamAutomaticMaskGenerator(sam_model, stability_score_thresh=0.85, points_per_side=16)
  unfiltered_mask_data = mask_generator.generate(np_image)

  # filter submasks and masks that can be broken up
  mask_data = filter_masks(unfiltered_mask_data)

  return mask_data, unfiltered_mask_data



def is_made_of_submasks(mask_data, k):
  """Tests if the kth mask is mostly covered by the union of submasks that are a significant proportion of this mask.
  If this is the case, it's probably better to use the submasks instead of this mask.
  """

  min_intersection_fraction = 0.1
  submask_union = np.full(mask_data[k]['segmentation'].shape, False)

  for i, mask_object in enumerate(mask_data):
    if i == k:
      continue

    # ignore bigger masks, probably not a submask
    if mask_object['area'] > mask_data[k]['area']:
      continue

    # ignore tiny masks
    if mask_object['area'] < min_intersection_fraction * mask_data[k]['area']:
      continue

    intersection = mask_object['segmentation'] & mask_data[k]['segmentation']
    if np.count_nonzero(intersection) / mask_data[k]['area'] > min_intersection_fraction:
      submask_union = submask_union | intersection

    # if submasks managed to cover an appreciable fraction of this mask, return true
    # from testing, "appreciable fraction" should actually be quite small
    if np.count_nonzero(submask_union) > 0.3*mask_data[k]['area']:
      return True

  # print(k, "Not covered enough - fraction covered:", np.count_nonzero(submask_union)/mask_data[k]['area'])
  return False



def filter_masks(mask_data):
  # go through masks biggest -> smallest, so we can filter out masks that are mostly in already covered areas (submasks of a larger mask)
  sorted_mask_data = sorted(mask_data, key=lambda x: x['area'], reverse=True)
  filtered_masks = []

  covered_area = np.full(mask_data[0]['segmentation'].shape, False)

  for i, mask_object in enumerate(sorted_mask_data):
    mask = mask_object['segmentation']

    # filter out masks that take up basically the whole image - this will be a zoom level anyways
    # this helps us still keep important smaller masks when we filter submasks out
    width_fraction = mask_object['bbox'][2] / mask.shape[1]
    height_fraction = mask_object['bbox'][3] / mask.shape[0]
    area_fraction = mask_object['area'] / mask_object['segmentation'].size
    if (width_fraction > 0.9 and height_fraction > 0.9) or area_fraction > 0.5:
      continue

    # filter out masks that are mostly made up of submasks
    if is_made_of_submasks(sorted_mask_data, i):
      continue
      
    # filter out if this is a submask
    mask_fraction_already_covered = np.count_nonzero(mask & covered_area) / np.count_nonzero(mask)
    if mask_fraction_already_covered > 0.5:
      continue

    filtered_masks.append(mask_object)
    covered_area = covered_area | mask

  return filtered_masks


def get_prominence(mask_object, inverse_depth):
  avg_inv_depth = np.mean(inverse_depth[mask_object['segmentation']]).astype('float64')
  area_fraction = mask_object['area'] / mask_object['segmentation'].size
  return (avg_inv_depth**2) * area_fraction



# Image classification using CLIP

def classify(np_image, clip_processor, clip_model, prompts):
    # image is a numpy image, convert to PIL
    image = Image.fromarray(np_image)

    logging.disable(logging.WARNING)  # get rid of annoying padding message
    inputs = clip_processor(text=prompts, images=image, return_tensors="pt", padding=True)  # specifying padding=True seems to help if there are spaces in your input
    logging.disable(logging.NOTSET)
    
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    return prompts[probs.argmax()]




# Bounding box utilities

def iou(box1, box2):
    # boxes are XYWH

    # extract coordinates and dimensions
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # calculate intersection area
    x_intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_intersection = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection_area = x_intersection * y_intersection

    # calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    # calculate IoU
    return intersection_area / union_area


def deduplicate_boxes(boxs, iou_threshold=0.5):
    # deduplicate list of boxs, using intersection over union to identify duplicates
    # boxs are XYWH
    # keep the larger box when there is a duplicate

    deduplicated_boxs = []
    
    for box in boxs:
        is_duplicate = False
        for i, existing_box in enumerate(deduplicated_boxs):
            iou_value = iou(box, existing_box)
            if iou_value >= iou_threshold:
                is_duplicate = True
                # keep the bigger of the two duplicates
                if box[2] * box[3] > existing_box[2] * existing_box[3]:
                    deduplicated_boxs[i] = box
                break

        if not is_duplicate:
            deduplicated_boxs.append(box)

    return deduplicated_boxs