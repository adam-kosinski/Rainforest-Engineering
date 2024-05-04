This repository contains the code for the ID from images team's project in Spring 2024 Rainforest Engineering. The final report associated with this code can be found here: https://docs.google.com/document/d/1cotW2eRiufb9mtZnNRY4hh2Nt7IbSlA7-Xfzl7O2HDE/edit#heading=h.8tqtb432dqt5.

## Installation

Note - this code is already set up on the Prostar gaming laptop that Dr. Brooke has for Rainforest Engineering / XPRIZE. The conda environment is called plant_id. The code is cloned in Documents/id_from_image.

To run the main image processing pipeline for finding plants, you will need an NVIDIA GPU or it will take forever. For preprocessing and running the UI, a GPU is not required.

### Installation

This installation mixes pip and conda because we were struggling to install some of this stuff using conda.

Clone this repository. Then run:

```
conda create --name plant_id python=3.10
conda activate plant_id
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

pip install matplotlib==3.8.0 opencv-python==4.9.0.80 scikit-learn==1.4.2 scikit-image==0.23.2 pandas==2.2.2 PyQt6==6.6.1 timm==0.9.16
```
Install these from source to avoid bugs:
```
pip install git+https://github.com/huggingface/transformers
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

In order to open/run jupyter notebooks, you might need to `pip install` some other stuff, but this may depend on your computer's setup. Try to run the notebook and if it doesn't let you, install what it tells you to.


### Google Colab
Same thing, except you probably don't need to install pytorch, since it is by default installed, and also things like matplotlib and pandas are probably already installed.

Feel free to reach out to me (adam.kosinski@duke.edu) if you are struggling to install this code.


## How To Run

See the implementation section of the report for a guide to the preprocessing scripts. Sorting images based on flower likelihood is demonstrated in `experiments/flower_search.ipynb`.

The main plant-finding script is `find_plants.py`. See `how_to_run.ipynb` for an example of how to use it.

To launch the user interface, run `python ui.py`.