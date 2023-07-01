from pymongo import MongoClient
import numpy as np

def connect():
    client = MongoClient('mongodb://nastava.is.pmf.uns.ac.rs:27017/')
    db = client['databases']

    collection = db['imageEmbeddings']

    return collection

def get_image_collection():
    images = connect().find()
    image_embeddings = []
    for image in images:
        embedding = image['embedding']
        image_embeddings.append(embedding)
    image_embeddings = np.array(image_embeddings)
    return image_embeddings

def insert_embedd(collection, image_name, embedding):
    embedding_doc = {
        "image_name": image_name,
        "embedding": embedding,
    }

    collection.insert_one(embedding_doc)
    print("Successfully saved!")

def find_document_by_embedding(embedding):
    query = {"embedding": embedding}
    collection = connect()
    document = collection.find_one(query)
    #only one document found and return
    return document