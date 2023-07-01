from pymongo import MongoClient
import numpy as np

def connect():
    client = MongoClient('mongodb://nastava.is.pmf.uns.ac.rs:27017/')
    db = client['databases']
    collection = db['imageDocuments']
    return collection

def get_image_collection():
    images = connect().find()
    image_embeddings = []
    for image in images:
        embedding = image['image_embedding']
        image_embeddings.append(embedding)
    image_embeddings = np.array(image_embeddings)
    return image_embeddings

def insert_embedd(collection, image_id, image_name, real_image, 
                  image_embedding, caption_to_use, text_embedding):
    image_doc = {
        "image_id": image_id,
        "image_name": image_name,
        "real_image": real_image,
        "image_embedding": image_embedding,
        "caption_to_use": caption_to_use,
        "text_embedding":text_embedding,

    }
    collection.insert_one(image_doc)
    #print("Successfully saved!")

def find_document_by_embedding(embedding):
    query = {"image_embedding": embedding}
    collection = connect()
    document = collection.find_one(query)
    #only one document found and return
    return document