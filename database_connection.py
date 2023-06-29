from pymongo import MongoClient

def connect():
    client = MongoClient('mongodb://nastava.is.pmf.uns.ac.rs:27017/')
    db = client['databases']  # Zamijenite 'ime_baze' sa stvarnim imenom baze

    collection = db['imageEmbeddings']

    return collection

def insert_embedd(collection, image_name, embedding):
    embedding_doc = {
        "image_name": image_name,
        "embedding": embedding,
    }

    collection.insert_one(embedding_doc)
    print("Successfully saved!")
