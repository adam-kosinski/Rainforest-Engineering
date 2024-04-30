import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import sys


def draw_segmented_objects(image, contours, label_cnt_idx):
    mask = np.zeros_like(image[:, :, 0])
    cv2.drawContours(mask, [contours[i] for i in label_cnt_idx], -1, (255), -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    # masked_image = cv2.putText(masked_image, f'{bubbles_count} bubbles', (200, 1200), cv2.FONT_HERSHEY_SIMPLEX,
    #                    fontScale = 3, color = (255, 255, 255), thickness = 10, lineType = cv2.LINE_AA)
    return masked_image


def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, _


def get_filtered_contours(image, contours):
    # create list for contours that are above a certain area
    filtered_contours = []

    # create dataframe to store mean color of each contour
    df_mean_color = pd.DataFrame()

    # iterate through all contours
    for idx, contour in enumerate(contours):
        # get area of contour
        area = int(cv2.contourArea(contour))

        # if area is higher than 3000:
        # takes longer to run if "area > 3000" taken out
        if area > 3000:
            # add contour to list of filtered contours
            filtered_contours.append(contour)

            # get mean color of contour:
            masked = np.zeros_like(
                image[:, :, 0]
            )  # This mask is used to get the mean color of the specific bead (contour), for kmeans

            # draw contour on mask
            cv2.drawContours(masked, [contour], 0, 255, -1)

            # get mean color of contour
            B_mean, G_mean, R_mean, _ = cv2.mean(image, mask=masked)

            # store in df
            df = pd.DataFrame(
                {"B_mean": B_mean, "G_mean": G_mean, "R_mean": R_mean}, index=[idx]
            )

            # append temp df to main df
            df_mean_color = pd.concat([df_mean_color, df])

    return df_mean_color, filtered_contours


def cluster(df_mean_color, n_clusters=3):
    km = KMeans(n_clusters=n_clusters)
    df_mean_color["label"] = km.fit_predict(df_mean_color)
    return df_mean_color


def segment(image, df_mean_color, contours, filename, iterator):
    img = image.copy()
    for label, df_grouped in df_mean_color.groupby("label"):
        good_flag = 0
        masked_image = draw_segmented_objects(image, contours, df_grouped.index)
        img = cv2.hconcat([img])
        # 'masked image' here are the actual clusters
        # get_edge_count(masked_image) # could add this back in as function - just wont implement as it doesnt really work

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to increase contrast
        equalized = cv2.equalizeHist(gray)

        # Calculate the mean pixel value of the equalized image
        mean_value = np.mean(equalized)

        # If the mean pixel value is above a certain threshold, consider the image "good"
        print(f"mean value: {mean_value}")
        if mean_value > 50:
            print("Good image detected - saving.")
            good_flag = 1

        else:
            print("Bad image detected.")

        if good_flag == 1:
            # export image
            cv2.imwrite(f"clustered/segmented_{filename}", img)

    # increase size of plot
    plt.figure(figsize=(20, 20))

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # increase size of image variable
    plt.figure(figsize=(20, 20))


def process_single_image(image, iterator):
    try:
        contours, _ = find_contours(image)

        # pass output to get_filtered_contours
        # function is currently only used to get df_mean_color - not doing anything with filtered contours
        df_mean_color, filtered_contours = get_filtered_contours(image, contours)

        df_mean_color = cluster(df_mean_color, n_clusters=3)

        # segmentation step
        # passing in filtered contours gives list index out of range error
        segment(image, df_mean_color, contours, filename, iterator=0)
    except:
        pass


def process_folder(folder):
    iterator = 0

    for f in os.listdir(folder):
        image_path = os.path.join(folder, f)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        process_single_image(image, iterator=iterator)
        iterator += 1


def get_filetype(filename):
    """Determines the type of a given file
    If the file is am image, it returns the image type
    If the file is a directory, it returns dir"""

    if os.path.isdir(filename):
        return "dir"
    elif os.path.isfile(filename):
        # split filename then return after .
        print(f"filetype is: {filename[-4:]}")

        if filename[-4:] == ".mp4":
            print("filetype is mp4 - need to run sample frames on this first")
            sys.exit(1)

        return filename[-4:]
    else:
        return None


if __name__ == "__main__":
    # if no folder called results/clustered exists, create it
    if not os.path.exists("clustered/"):
        os.makedirs("clustered/")

    # take first element in argv as image path
    if len(sys.argv) != 2:
        print("Incorrect number of arguments")
        sys.exit(1)

    image_path = sys.argv[1]
    filename = os.path.basename(image_path)
    print(f"filename is: {image_path}")

    # If a file is passed as an argument, get the file type. Will help us know what to do with the file
    if image_path:
        filetype = get_filetype(image_path)
        print(f"filetype is {filetype}")

        if filetype is None:
            print("File does not exist or file of wrong type passed")
            sys.exit(1)

        # print(sys.argv[1])  # debug
        # print(filetype)  # debug

    if filetype == "dir":
        print("Processing folder")
        process_folder(image_path)

    else:
        print("Processing single image")
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        process_single_image(image, iterator=0)
