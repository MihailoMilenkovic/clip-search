from dataset_loader import load_coco_dataset

for split in ["test","train"]:
    data=load_coco_dataset(split,batch_size=1,preprocessing_type="indexing")
    for (real_image,train_image,caption,image_filename,image_id) in data:
      caption_to_use=get_random_captions(caption)
      text_embedding=get_text_embedding(caption_to_use)
      image_embedding=get_image_embedding(train_image)
      print("saving data:",real_image,image_embedding,caption_to_use,text_embedding,image_filename,image_id)