from clip_model import CLIP
from dataset_loader import image_preprocessor
import database_connection as db
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import lucene_functions as luf

#ukoliko tako zelimo, mozemo zapravo da indeksiramo slike, samo ja nisam uspela PyLucene da uvedem u python :(
#mada mislim da ima preko dockera da se preuzme, pa da radi, stavila sam upustvo u txt fajl
#takodje sam u lucene_functions.py stavila kod za indeksiranje i pretragu
#TODO (@Ivana): finish everything below
def index_images():
  clip_model=CLIP()
  clip_model.load_model("models/CLIP")  
  #TODO: load images from some dataset (TBD which one) ??? Mihailo da vidi sta ce sa ovim
  images=[("image_name_1","contents1"),("image_name_2","contents2")]
  image_embeddings_csv=open("image_embeddings.csv","w")
  collection = db.connect() #make connection to database
  for (image_name,image_conents) in images:
    image_contents=image_preprocessor(image_contents)
    image_embedding=clip_model.getImageEmbedding(image_conents)
    #TODO: (DONE) save contents to database properly, use a database (probably mongodb or something similar)
    db.insert_embedd(collection, image_name, image_embedding) # saved to MongoDB database (fakultetska baza)
    #image_embeddings_csv.write(f"{image_name},{image_embedding}\n")

  #Ovde bi onda nakon sto smo sacuvali sve podatke u bazu, trebalo da ih indeksiramo 
  luf.images_indexing(collection.find())

def get_k_most_similar_images(text_embedding, image_embeddings, k):
  #TODO: query database for most similar image using HNSW algorithm, use l2 distance for similarity
  #TODO: also have function to return top k most similar images, possibly also to sort whole database by similarity?
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
    return image_doc #return image doc which continues the closest embedding

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
    return image_doc #return image doc which continues the closest embedding

def get_most_similar_image(text_embedding, image_embeddings, k):
  return get_k_most_similar_images(text_embedding, image_embeddings, k)[0]


def find_most_similar_image(user_text_input):
  clip_model=CLIP()
  clip_model.load_model("models/CLIP")
  text_embedding=clip_model.getTextEmbedding(user_text_input)
  #TODO: load everything from database properly done 
  image_collection = db.get_image_collection() #actually returns image embeddings from database
  result_image_name=get_most_similar_image(text_embedding, image_collection, 1)
  result_image_contents=result_image_name
  return result_image_contents