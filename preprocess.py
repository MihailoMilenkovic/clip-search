import random
from PIL import Image
import os
import numpy as np
import json

from pycocotools.coco import COCO

image_source_dir="train2014"
image_dest_dir="preprocessed"
annotation_dir="annotations"
metadata_dir="metadata"

caption_file=f"{annotation_dir}/instances_train.json"
coco_caps=COCO(caption_file)

def preprocess_image(img,img_average=None):
  #resizing so that smaller dimension is from 256 to 480px
  #then subtracting means from each pixel
  #then resizing to 224x224
  if img_average==None:
    img_average=Image.open(f"{metadata_dir}/average")
  s_min=256
  s_max=480
  crop_size=224
  img=img-img_average
  img_size=img.size
  s=random.randrange(s_min,s_max+1)
  resizing_ratio=s/img_size
  new_size=(int(img_size[0]*resizing_ratio()),int(img_size[1]*resizing_ratio))
  img=img.resize(new_size)
  img_starts=(random.randrange(0,new_size[0]-crop_size),random.randrange(0,new_size[1]-crop_size))
  img=img.crop(img_starts[0],img_starts[1],img_starts[0]+crop_size-1,img_starts[1]+crop_size-1)
  return img


def byte_pair_encode_dataset():
  #TODO: parse dataset using BPE with sentencepiece (https://github.com/google/sentencepiece) and store labels
  #TODO: store information such as vocab_size in metadata_dir
  pass

vocab_size=30
def get_byte_pair_encoding(text):
  #TODO: get stored byte pair encoding and encode text
  return text

def get_batch_annotated_images(batch_size=8):
  image_names=os.listdir(image_source_dir)
  image_names_selected=random.choices(image_names,k=batch_size)
  image_batch=[]
  caption_batch=[]
  for image_name in image_names_selected:
    image=Image.open(f"{image_dest_dir}/{image_name}")
    image_batch.append[image]
    image_captions = coco_caps.getAnnIds(imgIds=image_name)
    if len(image_captions)>0:
      caption=image_captions[random.randint(0,len(image_captions)-1)]
      caption=get_byte_pair_encoding(caption)
      caption_batch.append(caption)
  return image_batch,caption_batch

if __name__=="__main__":
  image_names=os.listdir(image_source_dir)
  example_image=Image.open(f"{image_source_dir}/{image_names[0]}")
  img_sum=np.zeros(shape=example_image.shape,dtype=np.float32)
  img_num=len(image_names)
  for image_name in image_names:
    img=Image.open(image_name)
    img_sum+=img
  img_average=img_sum/img_num
  img_average=np.round(img_average).astype(np.uint8)
  img_average=Image.fromarray(img_average)
  img.save(f"{metadata_dir}/average")

  for image_name in os.listdir(image_source_dir):
    img=Image.open(image_name)
    img=preprocess_image(img,img_average)
    img.save(f"{image_dest_dir}/{image_name}")
