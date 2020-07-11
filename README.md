# DETECTION OF PROPAGANDA TECHNIQUES IN NEWS ARTICLES
This repository contains code for SEMEVAL 2020 TASK 11 "DETECTION OF PROPAGANDA TECHNIQUES IN NEWS ARTICLES." Our system was ranked 5th for the Technique Classification subtask. 
The source code is available in the [src](https://github.com/paramansh/propaganda_detection/tree/master/src) directory. 

Note that this code uses version 2 of transformers library (2.8.0)

## Task SI
### Training 
`$ python src/train_SI.py`

The trained models will be stored in [model_dir](https://github.com/paramansh/propaganda_detection/tree/master/model_dir). Note that currently we save model if number of trainining articles are greater than 150 (This can be changed [here](https://github.com/paramansh/propaganda_detection/blob/0f6e4e478edbfd88efce1ea03f674b406c0a0d9e/src/train_SI.py#L137))
### Prediction
`$ python get_outputs.py`

This outputs the predicted spans for article in `input.txt`. The predicted spans are shown in bold.

If the model is not present in `model_dir`, it downloads our BERT_BIOE model avalaible [here](https://drive.google.com/uc?id=1-5oN2lS37IcXT1Lhd-H3TxlEdi4MzVPC) and stores it in `model_dir`. To make prediction for any custom model, change the model path in `src/predict_SI.py`

To get outputs for dev-set (or any other folder)
`$ python get_outputs.py --dev-output True`

## Task SI
### Training 
`$ python src/train_SI.py`

The trained models will be stored in [model_dir](https://github.com/paramansh/propaganda_detection/tree/master/model_dir).
### Prediction
`$ python src/predict_TC.py`
This outputs the techniques of given spans.


## Colab Notebooks
Interactive iPython notebooks for both tasks and related experiments can be found
in [Colab Notebooks](https://github.com/paramansh/propaganda_detection/tree/master/Colab%20Notebooks) directory.
