import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np

import tensorflow_datasets as tfds

def preprocess_data_for_model(example):
  image_preprocessor=keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(224, 224),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
  )
  image=image_preprocessor(example["image"])
  captions=example["captions"]["text"]
  return image,captions

def preprocess_data_for_indexing(example):
  real_image=example["image"]
  image_filename=example["image/filename"]
  image_id=example["image/id"]
  train_image,captions=preprocess_data_for_model(example)
  return real_image,train_image,captions,image_filename,image_id

#when training, use "training" preprocessing type to get only relevant data preprocessed in correct format
#when indexing, use "indexing" preprocessing type to get actual images as well as preprocessed ones to use as input to model  
def load_coco_dataset(split="train", batch_size=8, preprocessing_type="training"):
  assert split in ["train","test"]
  assert preprocessing_type in ["training","indexing"]
  dataset, info = tfds.load("coco_captions",split=split,with_info=True)
  # max_len=len(dataset)
  dataset=dataset.filter(lambda example: len(example["captions"]["text"])==5)
  preprocessing_func=preprocess_data_for_model if preprocessing_type=="training" else preprocess_data_for_indexing
  dataset=dataset.map(preprocessing_func, num_parallel_calls=tf.data.AUTOTUNE)
  dataset=dataset.shuffle(100).batch(batch_size)
  dataset=dataset.take(100) #TODO: remove limit after testing on smaller dataset
  return dataset
