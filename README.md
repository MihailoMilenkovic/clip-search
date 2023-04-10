# Clip search

This repository contains code for jointly training a vision and language model using the CLIP objective.\
The models trained are a vision transformer and a text transformer and training is done on the COCO dataset (https://cocodataset.org/#home)\
This repository also contains a website where users can submit text or image queries.\
The model returns the image with the most similar CLIP embedding as well as its and corresponding description 

## Training the model

```sh
python train.py
```

## Running the server
```sh
python server.py
```