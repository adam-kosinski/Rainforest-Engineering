import sys
from preprocess_util import process_images, save_to_csv, find_similar_and_log

def process_folder(folder_path, output_folder="./final_preprocessed"):
    """
    Processes images in the given folder, saves laplacian variances to a CSV,
    and finds similar images to log them in another CSV file.

    :param folder_path: The path to the folder containing images to process.
    """
    print(f"Processing folder: {folder_path}")
    process_images(folder_path, output_folder, threshold=500)
    find_similar_and_log(output_folder, threshold=0.8)

if __name__ == "__main__":
    # Check if the input path is provided; if not, use './clustered' as the default folder path
    if len(sys.argv) < 2:
        folder_path = "./clustered"
        print(f"No path provided. Using default path: {folder_path}")
    else:
        folder_path = sys.argv[1]

    process_folder(folder_path)