import tensorflow as tf
from pymongo import MongoClient
import numpy as np
import bson

def connect(col_name):
    # mongo_conn_str='mongodb://nastava.is.pmf.uns.ac.rs:27017/'
    mongo_conn_str='mongodb://localhost:27017/'
    client = MongoClient(mongo_conn_str)
    db = client['databases']
    collection = db[col_name]
    return collection

def get_image_collection():
    images = connect('imageDocuments').find()
    image_embeddings = []
    for image in images:
        embedding = image['image_embedding']
        image_embeddings.append(np.array(embedding)[0])
    #image_embeddings = np.array(image_embeddings)
    return np.array(image_embeddings)

def insert_embedd(collection, image_id, image_name, real_image, 
                  image_embedding, caption_to_use, text_embedding):
                  
    image_id_to_save=int(image_id[0].numpy())
    image_name_to_save=str(image_name[0].numpy())
    caption_to_save=str(caption_to_use[0][0].numpy())
    image_embedding_to_save=image_embedding.numpy().tolist()
    image_to_save=bson.Binary(real_image)
    text_embedding_to_save=text_embedding.numpy().tolist()
    print("saving data:------------------")
    image_doc = {
        "image_id": image_id_to_save,
        "image_name": image_name_to_save,
        "real_image":image_to_save , #storage real image as binary data
        "image_embedding": image_embedding_to_save, #from tensor to ndarray and to list embedding 
        "caption_to_use": caption_to_save,
        "text_embedding": text_embedding_to_save, #from tensor to ndarray  and to list embedding
    }
    collection.insert_one(image_doc)
    #print("Successfully saved!")

def find_document_by_embedding(embedding):
    collection = connect('imageDocuments')
    query = {"image_embedding": embedding.tolist()}
    document = collection.find_one(query)
    #only one document found and returned
    return document