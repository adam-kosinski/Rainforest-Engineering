import subprocess
import glob

def process_path(path):
    """
    Processes a given path by calling a sample frames script,
    and then passes every file in the results subfolders to a clustering script.

    :param path: The input path to process.
    """
    print(f"path is {path}")

    # Pass this path to sample_frames.py
    subprocess.run(["python", "sample_frames.py", path], check=True)

    # Pass every file in the results subfolders to clustering.py
    for file_path in glob.glob('results/*/*', recursive=True):
        subprocess.run(["python", "clustering.py", file_path], check=True)

if __name__ == "__main__":
    import sys

    # Ensure there is an input path provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <path>")
        sys.exit(1)

    input_path = sys.argv[1]
    process_path(input_path)