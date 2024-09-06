import glob
import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from transformers import AutoTokenizer
from torch import cuda
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from auditqa.reports import files, report_list
from langchain.docstore.document import Document
import configparser

# read all the necessary variables
device = 'cuda' if cuda.is_available() else 'cpu'
path_to_data = "./reports/"       


##---------------------fucntions -------------------------------------------##
def getconfig(configfile_path:str):
    """
    configfile_path: file path of .cfg file
    """

    config = configparser.ConfigParser()

    try:
        config.read_file(open(configfile_path))
        return config
    except:
        logging.warning("config file not found")
        
def open_file(filepath):
    with open(filepath) as file:
        simple_json = json.load(file)
    return simple_json

def load_chunks():
    """
    this method reads through the files and report_list to create the vector database
    """

    #  we iterate through the files which contain information about its
    # 'source'=='category', 'subtype', these are used in UI for document selection
    #  which will be used later for filtering database
    config = getconfig("./model_params.cfg")
    all_documents = {}
    categories = list(files.keys())
    # iterate through 'source'
    for category in categories:
        print("documents splitting in source:",category)
        all_documents[category] = []
        subtypes = list(files[category].keys())
        # iterate through 'subtype' within the source
        # example source/category == 'District', has subtypes which is district names
        for subtype in subtypes:
            print("document splitting for subtype:",subtype)
            for file in files[category][subtype]:

                # load the chunks
                try:
                    doc_processed = open_file(path_to_data + file + "/"+ file+ ".chunks.json" )

                
                except Exception as e:
                    print("Exception: ", e)
                print("chunks in subtype:",subtype, "are:",len(doc_processed))

                # add metadata information 
                chunks_list = []
                for doc in doc_processed:
                    chunks_list.append(Document(page_content= doc['content'], 
                             metadata={"source": category,
                                      "subtype":subtype,
                                      "year":file[-4:],
                                      "filename":file,
                                      "page":doc['metadata']['page'],
                                      "headings":doc['metadata']['headings']}))

                all_documents[category].append(chunks_list)
    
    # convert list of list to flat list
    for key, docs_processed in all_documents.items():
        docs_processed = [item for sublist in docs_processed for item in sublist]
        print("length of chunks in source:",key, "are:",len(docs_processed))
        all_documents[key] = docs_processed
    all_documents['allreports'] = [sublist for key,sublist in all_documents.items()]
    all_documents['allreports'] = [item for sublist in all_documents['allreports'] for item in sublist]
    # define embedding model
    embeddings = HuggingFaceEmbeddings(
        model_kwargs = {'device': device},
        encode_kwargs = {'normalize_embeddings': bool(int(config.get('retriever','NORMALIZE')))},
        model_name=config.get('retriever','MODEL')
    )
    # placeholder for collection
    qdrant_collections = {}
    
    
    for file,value in all_documents.items():
        if file == "allreports":
            print("emebddings for:",file)
            qdrant_collections[file] = Qdrant.from_documents(
                value,
                embeddings,
                location=":memory:", 
                collection_name=file,
            )
    print(qdrant_collections)
    print("vector embeddings done")
    return qdrant_collections