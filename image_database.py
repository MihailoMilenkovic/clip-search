from clip_model import CLIP
from dataset_loader import image_preprocessor
import database_connection as db
import numpy as np
from sklearn.neighbors import NearestNeighbors
import lucene_functions as luf

#ovde treba zapravo da indeksiramo slike, samo ja nisam uspela PyLucene da uvedem u python :(
#mada mislim da ima preko dockera da se preuzme, pa da radi, stavila sam upustvo u txt fajl
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

def get_k_most_similar_images(text_embedding, k):
  #TODO: query database for most similar image using HNSW algorithm, use l2 distance for similarity
  #TODO: also have function to return top k most similar images, possibly also to sort whole database by similarity?
  collection = db.connect() #make connection
  #here is basic knn
  # get all embeddings from database
  embeddings = []
  for document in collection.find():
    embedding = document['embedding']
    embeddings.append(embedding)
  embeddings = np.array(embeddings)
  knn = NearestNeighbors(n_neighbors=k, metric='cosine') #mozemo menjati metriku, Euclidian, Cosine, Manhattan..
  knn.fit(embeddings)

  distances, indices = knn.kneighbors([text_embedding])
  """
  for distance, index in zip(distances[0], indices[0]):
    result = collection.find_one({'embedding': embeddings[index]})
    print(': {:.2f}, Document: {}'.format(distance, result))
    """

def get_most_similar_image(text_embedding):
  return get_k_most_similar_images(text_embedding,1)[0]


def find_most_similar_image(user_text_input):
  clip_model=CLIP()
  clip_model.load_model("models/CLIP")
  text_embedding=clip_model.getTextEmbedding(user_text_input)
  result_image_name=get_most_similar_image(text_embedding)
  #TODO: load everything from database properly ??
  result_image_contents=result_image_name
  return result_image_contents