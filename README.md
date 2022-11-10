# sausage-classifier
A classification model toolkit to identify sausages. 

![Sausage](https://live.staticflickr.com/499/32113273682_effd1084a6_b.jpg)

## Install
Clone the repo and install the requirements.

    pip install -r requirements.txt
    
## Training
    
Run the training.

    python train.py 
    
If this is your first time, training data will automatically be webscraped and organised into the necessary file structure.

    - input
        - train
            - sausage
            - non-sausage
        - valid
            - sausage
            - non-sausage
            
            
### Training Options

#### Model

Specify which model you'd like to train using the `--model` flag. There are currently two options:
- `CNN`: A basic CNN architecture
- `ViT`: A pretrained [ViT (Vision Transformer) model](https://huggingface.co/docs/transformers/model_doc/vit) 
from HuggingFace, fine-tuned on local training data

This defaults to the `CNN` model.

    python train.py --model ViT
            
#### Epochs

Specify the number of training epochs with the optional `--epochs` flag. This defaults to 20 epochs. 

    python train.py --epochs 35
    
    
#### Training Data
    
Force the image web-scraper to rerun by setting the optional `--redownload` flag to True.

    python train.py --redownload True
    
This is useful if you've amended the image search queries. These are found in the `image_queries.json` file and 
can be updated.

Currently 20 images are scraped for each query, 15 of which are placed in `train/` and 5 are placed in `valid/`.


## Inference

Run the model on one given image and see its prediction by running `inference.py` with a filepath. 
If no path is provided, this defaults to the first image file in the `input/valid/sausage/` directory.

    python inference.py input/valid/sausage/chosen_image.jpg

This returns the specified image overlaid with the ground_truth (GT) and model predictions. 
Work will be done to make this look nicer. 

### Inference Options

#### Model

Specify which model you'd like to run inference on by setting the `--model` flag to the desired `.pth` file. These 
files are found in the `outputs/` directory. 

This defaults to the first `.pth` file in the directory. 

    python inference.py input/valid/sausage/chosen_image.jpg --model CNNModel_model_20_epochs.pth

#### Show Image

Turn off the function to display the chosen image by setting the `--display` flag to `False`. This defaults to `True`. 

    python inference.py input/valid/sausage/chosen_image.jpg --display False


## To-Do
- Add more models to compare
- Add more performance metrics
- ~~Add args to allow users to select models~~
- ~~Add greater flexibility when saving models/outputs~~
- ~~Add model flexibility in `inference.py`~~
- Improve inference.py output image
- ~~Explore alternative webscraping for training data - larger images etc~~
- ~~Webscraping queries read from json/txt file, args to specify path~~
- Explore active learning for scraping additional images
- Update powershell script to bash 
- imwrite() in `inference.py`
- ~~Randomise test/train split~~
- ~~Create branch training on multiple classes~~
  - ~~Alter directory structure when webscraping~~
  - ~~Adjust classes in models~~
- ~~Read labels/num_labels from .json rather than hardcoding~~
- Automate finding and removing near-duplicate images from training data ([resource](https://towardsdatascience.com/find-and-remove-duplicate-images-in-your-dataset-3e3ec818b978))
- Add logging
- Add fiftyone labelling capability


## Notes 
- The image query 'sausage plate' was polluting the training/validation data with chopped sausages, wellingtons etc. 
Replacing this query with 'single sausage' improved average accuracy. 
- It's possible that the 'quality' of search images degrades the further down the list you go. In other words, the top 
Google search image may be more accurate than the 20th. To account for this, a random shuffle was added after the images
are webscraped and before they are written to files. 
- Accuracy greatly improved when Shutterstock images were added as well as the existing Google images. 
The model was improved by a larger training set (with potentially larger/higher quality images)
