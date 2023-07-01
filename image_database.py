from dataset_loader import image_preprocessor
import database_connection as db
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from dataset_loader import load_coco_dataset
from clip_model import CLIP

#TODO (@Ivana): finish everything below
def index_images():
  batch_size=1
  model=CLIP(batch_size=batch_size)
  model.load_model()
  for split in ["test","train"]:
    data=load_coco_dataset(split,batch_size=batch_size,preprocessing_type="indexing")
    collection = db.connect() #make connection to database
    for (real_image,train_image,captions,image_filename,image_id) in data:
      caption_to_use=model.get_random_captions(captions)
      text_embedding=model.get_text_embedding(caption_to_use)
      image_embedding=model.get_image_embedding(train_image)
      #print("saving data:",real_image,image_embedding,caption_to_use,text_embedding,image_filename,image_id)
      # saved to MongoDB database (fakultetska baza)
      db.insert_embedd(collection, image_id, image_filename, real_image, image_embedding, caption_to_use, text_embedding)
      #break

def get_k_most_similar_images(text_embedding, image_embeddings, k):
  #here is basic knn
  knn = NearestNeighbors(n_neighbors=k, metric='cosine') #mozemo menjati metriku, Euclidian, Cosine, Manhattan..
  knn.fit(image_embeddings)
  distances, indices = knn.kneighbors([text_embedding])
  """
  for distance, index in zip(distances[0], indices[0]):
    result = collection.find_one({'embedding': embeddings[index]})
    print(': {:.2f}, Document: {}'.format(distance, result))
    """
  
#maybe better version of implementing kNN
def get_k_most_similar_images_sec(text_embedding, image_embeddings, k):
    # Compute cosine similarity between text and every image in database
    similarities = cosine_similarity(text_embedding[np.newaxis, :], image_embeddings)
    # K nearest images:
    nearest_indices = np.argsort(similarities[0])[::-1][:k] #find the closest embedding
    image_doc = db.find_document_by_embedding(nearest_indices)
    return image_doc #return image doc which contains the closest embedding

  #TODO: query database for most similar image, use l2 distance for similarity
  #TODO: also have function to return top k most similar images
def get_k_most_similar_images_L2(query_embedding, image_embeddings, k=5):
    computed_distances = []
    # Computed distances by Euclidian (L2) distance
    for i, embedding in enumerate(image_embeddings):
        distance = np.linalg.norm(query_embedding - embedding)
        computed_distances.append((i, distance))
    
    # Sort image embeddings according to distance
    computed_distances.sort(key=lambda x: x[1])
    
    # K neairest embeddings
    nearest_indices = [index for index, _ in computed_distances[:k]] 
    image_doc = db.find_document_by_embedding(nearest_indices)
    return image_doc #return image doc which contains the the closest embedding

def get_most_similar_image(text_embedding, image_embeddings, k):
  return get_k_most_similar_images(text_embedding, image_embeddings, k)["image_name"]

def find_most_similar_image(user_text_input):
  clip_model=CLIP()
  clip_model.load_model("models/CLIP")
  text_embedding=clip_model.getTextEmbedding(user_text_input)
  #TODO: load everything from database properly done 
  image_collection = db.get_image_collection() #actually returns only image embeddings from database
  #HERE TRY TO BUILD KD TREE FOR EFFICIENCY KNN
  result_image_name=get_most_similar_image(text_embedding, image_collection, 1)
  result_image_contents=result_image_name
  return result_image_contents