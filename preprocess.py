import random
from PIL import Image
import os
import numpy as np

image_source_dir="train2014"
image_dest_dir="preprocessed"
def preprocess_image(img,img_average=None):
  #resizing so that smaller dimension is from 256 to 480px
  #then subtracting means from each pixel
  #then resizing to 224x224
  if img_average==None:
    img_average=Image.open(f"{image_dest_dir}/average")
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
  img.save(f"{image_dest_dir}/average")

  for image_name in os.listdir(image_source_dir):
    img=Image.open(image_name)
    img=preprocess_image(img,img_average)
    img.save(f"{image_dest_dir}/{image_name}")
