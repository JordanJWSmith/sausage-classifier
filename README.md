# sausage-classifier
A simple classification model to identify sausages. 

![Sausage](https://live.staticflickr.com/499/32113273682_effd1084a6_b.jpg)

## Install
Clone the repo and install the requirements.

    pip install -r requirements.txt
    
    
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
            
#### Epochs

Specify the number of training epochs with the optional `--epochs` flag. This defaults to 20 epochs. 

    python train.py --epochs 35
    
    
#### Training Data
    
Force the image web-scraper to rerun by setting the optional `--redownload` flag to True.

    python train.py --redownload True
    
This is useful if you've amended the image search queries. These are found in the `image_queries.json` file and 
can be updated.

Currently 20 images are scraped for each query, 15 of which are placed in `train/` and 5 are placed in `valid/`.


### Inference

Run the model on one given image and see its prediction by running `inference.py` with a filepath. This filepath 
defaults to `sausage_16.jpg`.

    python inference.py input/valid/sausage/chosen_image.jpg

This returns the specified image overlaid with the ground_truth (GT) and model predictions. 
Work will be done to make this look nicer. 



## To-Do
- Add more models to compare
- ~~Add args to allow users to select models~~
- ~~Add greater flexibility when saving models/outputs~~
- Improve inference.py output image
- Explore alternative webscraping for training data - larger images etc
- ~~Webscraping queries read from json/txt file, args to specify path~~
- Explore active learning for scraping additional images
- Update powershell script to bash 
