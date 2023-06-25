from clip_model import CLIP

#TODO (@Vladimir): finish everything below
def index_labels():
  clip_model=CLIP()
  clip_model.load_model("models/CLIP")  
  #TODO: load labels from some dataset (TBD which one, probably imagenet/COCO classes, maybe pascal voc)
  labels=["label_1","label_2"]
  label_embeddings_csv=open("image_embeddings.csv","w")
  for (image_name,image_conents) in labels:
    image_embedding=clip_model.getImageEmbedding(image_conents)
    #TODO: save contents to database properly, see example in image_database.py
    label_embeddings_csv.write(f"{image_name},{image_embedding}\n")

def get_most_similar_class(image_embedding):
  #TODO: see image_database.py, use same approach 
  #test git
  pass

def find_most_similar_class(user_image_input):
  clip_model=CLIP()
  clip_model.load_model("models/CLIP")
  text_embedding=clip_model.getTextEmbedding(user_image_input)
  result_label=get_most_similar_class(text_embedding)
  return result_label

#TODO: iterate over dataset and get predicted and real class for each image
#TODO: calculate accuracy, etc.