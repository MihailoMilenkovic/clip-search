import database_connection as db
import numpy as np
from dataset_loader import load_coco_dataset
from clip_model import CLIP
import kd_tree as kd
import knn_impl as knn

#TODO (@Ivana): finish everything below
def index_images():
  batch_size=1
  model=CLIP(batch_size=batch_size)
  model.load_model()
  collection = db.connect('imageDocuments') #make connection to database
  for split in ["test","train"]:
    data=load_coco_dataset(split,batch_size=batch_size,preprocessing_type="indexing")
    for (real_image,train_image,captions,image_filename,image_id) in data:
      caption_to_use=model.get_random_captions(captions)
      text_embedding=model.get_text_embedding(caption_to_use)
      image_embedding=model.get_image_embedding(train_image)
      #print("saving data:",real_image,image_embedding,caption_to_use,text_embedding,image_filename,image_id)
      # saved to MongoDB database (fakultetska baza)
      db.insert_embedd(collection, image_id, image_filename, real_image, image_embedding, caption_to_use, text_embedding)
  collection_kd_tree = db.connect('imageKDTree')
  image_embeddings = db.get_image_collection() #list of embeddings of type ndarray
  kd_tree = kd.build_tree(image_embeddings)   #Build KD tree
  searilized_tree = kd.serialize_tree(kd_tree)
  collection_kd_tree.insert_one(searilized_tree)   #only one time save the tree in MongoDB

#knn algorithm on KD tree using euclidian distance
def get_k_most_similar_images(query_embedding, kd_tree, k):
  #search for the most similar embedding in tree
  the_closest_image_embedd = knn.knn_algorithm_euclidian(query_embedding, kd_tree, k) 
  return the_closest_image_embedd

def get_most_similar_image_L2(query_embedding, kd_tree, k):
  return get_k_most_similar_images(query_embedding, kd_tree, k)

#get most similar image using cosine metric
def get_most_similar_image_cosine(query_embedding, kd_tree, k):
  the_closest_image_embedd = knn.knn_search_cosine(query_embedding, kd_tree, k) 
  return the_closest_image_embedd

def find_most_similar_image(user_text_input, kd_tree):
  clip_model=CLIP()
  clip_model.load_model("models/CLIP")
  query_embedding=clip_model.getTextEmbedding(user_text_input)
  k = 1 #hyperparameter for kNN
  #result_image_embedding=get_most_similar_image_L2(query_embedding, kd_tree, k) #using Euclidian distance
  result_image_embedding=get_most_similar_image_cosine(query_embedding.numpy(), kd_tree, k) #using cosine similarity
  image_doc = db.find_document_by_embedding(result_image_embedding) 
  return image_doc['real_image'], image_doc['caption_to_use']

if __name__=="__main__":
  index_images()