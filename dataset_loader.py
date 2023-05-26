from pycocotools.coco import COCO
import numpy as np
import os
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers



# coco dataset loaded into coco folder by download script
coco_annotations_file = 'coco/annotations/instances_train.json'
coco_images_dir = 'coco/images'

# Initialize COCO instance for annotations
coco = COCO(coco_annotations_file)

# Get image IDs from COCO dataset
image_ids = coco.getImgIds()

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

# Map function to load and preprocess the images and load one image,annotation pair
def load_image_and_annotation(image_id):
  # Load image
  image_info = coco.loadImgs(image_id)[0]
  image_path = os.path.join(coco_images_dir, image_info['file_name'])
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = image_preprocessor(image)
  
  # Load annotations
  annotation_ids = coco.getAnnIds(imgIds=image_id)
  coco_annotations = coco.loadAnns(annotation_ids)
  used_annotation = coco_annotations[np.random.choice(coco_annotations)]
  caption = used_annotation['caption']
  
  return image, caption


def load_coco_dataset(num_batches=100, batch_size=8):
  dataset = tf.data.Dataset.from_tensor_slices(image_ids)
  dataset = dataset.map(load_image_and_annotation, num_parallel_calls=tf.data.AUTOTUNE)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat()
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  return dataset.take(num_batches)
