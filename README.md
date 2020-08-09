# Text extraction from Receipts using Deep Vision

Project repository for the masters course Deep Vision at the University of Heidelberg.

## Objective
In this project we implemented a pipeline that allows to extract all the text written on a receipt. For this we tested 
different deep learning architectures.   
For the text detection part the EAST model is used. For the text recognition we implemented a CRNN and a Dense Net as 
well as a combination of both.


## Usage 
To run the text prediction of a receipt do the following:  
```shell script
python receipt_pipeline.py path_to_image
``` 
This script will use a pretrained `EAST` and `CRNN` model stored in `trained_models` for the extraction of the text. If 
you want to use a different model run
```shell script
python receipt_pipeline.py -h
```
to see available options.