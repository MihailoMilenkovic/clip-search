from clip_model import CLIP
from dataset_loader import image_preprocessor
#TODO (@Ivana): finish everything below
def index_images():
  clip_model=CLIP()
  clip_model.load_model("models/CLIP")  
  #TODO: load images from some dataset (TBD which one)
  images=[("image_name_1","contents1"),("image_name_2","contents2")]
  image_embeddings_csv=open("image_embeddings.csv","w")
  for (image_name,image_conents) in images:
    image_contents=image_preprocessor(image_contents)
    image_embedding=clip_model.getImageEmbedding(image_conents)
    #TODO: save contents to database properly, use a database (probably mongodb or something similar)
    image_embeddings_csv.write(f"{image_name},{image_embedding}\n")


def get_k_most_similar_images(text_embedding,k):
  #TODO: query database for most similar image using HNSW algorithm, use l2 distance for similarity
  #TODO: also have function to return top k most similar images, possibly also to sort whole database by similarity?
  pass

def get_most_similar_image(text_embedding):
  return get_k_most_similar_images(text_embedding,1)[0]


def find_most_similar_image(user_text_input):
  clip_model=CLIP()
  clip_model.load_model("models/CLIP")
  text_embedding=clip_model.getTextEmbedding(user_text_input)
  result_image_name=get_most_similar_image(text_embedding)
  #TODO: load everything from database properly
  result_image_contents=result_image_name
  return result_image_contents