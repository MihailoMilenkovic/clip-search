import lucene
from org.apache.lucene.document import Document, Field, StringField, TextField
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import RAMDirectory
from org.apache.lucene.util import Version
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser

def images_indexing(images_data):
    # Inicialization of PyLucene
    lucene.initVM()
    #index directory
    directory = RAMDirectory()
    analyzer = StandardAnalyzer()
    # Index Writer
    config = IndexWriterConfig(analyzer)
    writer = IndexWriter(directory, config)
    # Iterirate through images and index them
    for document in images_data:
        image_name = document['image_name']
        embedding = document['embedding']
        # Create lucene document and add fields
        image_document = Document()
        image_document.add(StringField('image_name', image_name, Field.Store.YES))
        image_document.add(TextField('embedding', embedding, Field.Store.YES))

        # Add document to index
        writer.addDocument(image_document)

    # Close the writer
    writer.close()
    return directory

def search_for_image(query, directory):
    # Reader of index directory
    reader = DirectoryReader.open(directory)
    # Searcher
    searcher = IndexSearcher(reader)
    analyzer = StandardAnalyzer()

    # Query for search
    query = QueryParser(Version.LUCENE_CURRENT, 'embedding', analyzer).parse(query)
    top_images = searcher.search(query, 5)  # first 5 results
    score_images = top_images.scoreDocs

    # Score of the results
    for score_doc in score_images:
        doc_id = score_doc.doc
        lucene_image = searcher.doc(doc_id)
        # Take the info from the results
        image_name = lucene_image.get('image_name')
        embedding = lucene_image.get('embedding')
        print("Name of the image: "+image_name)
    reader.close()
