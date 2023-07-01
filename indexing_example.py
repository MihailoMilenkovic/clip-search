from dataset_loader import load_coco_dataset
from clip_model import CLIP

if __name__=="__main__":
  batch_size=1
  model=CLIP(batch_size=batch_size)
  model.load_model()
  for split in ["test","train"]:
    data=load_coco_dataset(split,batch_size=batch_size,preprocessing_type="indexing")
    for (real_image,train_image,caption,image_filename,image_id) in data:
      caption_to_use=get_random_captions(caption)
      text_embedding=model.get_text_embedding(caption_to_use)
      image_embedding=model.get_image_embedding(train_image)
      print("saving data:",real_image,image_embedding,caption_to_use,text_embedding,image_filename,image_id)