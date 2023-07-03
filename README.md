# Clip search

This repository contains code for jointly training a vision and language model using the CLIP objective.\
The models trained are a vision transformer and a text transformer and training is done on the COCO dataset (https://cocodataset.org/#home)\ This repository also contains a website where users can submit text or image queries.\
For text inputs, the image with the most similar CLIP embedding from the dataset is returned.\
For image inputs, the model finds the imagenet class label with the most similar CLIP embedding and returns it.

## Model training

- To train the model on the coco-captions dataset, run the following:

```sh
python train.py
```

## Running the server

- To index the database using the trained model and run the server, run the following:

```sh
docker run -d -p 27017:27017 --name m1 mongo
python index_image_database.py
```

```sh
docker build -t clip-server .
docker -p 8050:8050 clip-server
```
