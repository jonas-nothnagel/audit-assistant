import glob
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from transformers import AutoTokenizer
from torch import cuda
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from auditqa.reports import files, report_list
device = 'cuda' if cuda.is_available() else 'cpu'

### This script is NO MORE IN USE ##### 
# Preprocessed report pdf is brought along with chunks and added to existing reports database

# path to the pdf files
path_to_data = "./data/pdf/"

def process_pdf():
    """
    this method reads through the files and report_list to create the vector database
    """
    # load all the files using PyMuPDFfLoader
    docs = {}
    for file in report_list:
        try:
            docs[file] = PyMuPDFLoader(path_to_data + file + '.pdf').load()
        except Exception as e:
            print("Exception: ", e)

    
    # text splitter based on the tokenizer of a model of your choosing
    # to make texts fit exactly a transformer's context window size
    # langchain text splitters: https://python.langchain.com/docs/modules/data_connection/document_transformers/
    chunk_size = 256
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5"),
            chunk_size=chunk_size,
            chunk_overlap=10,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n"],
    )
    #  we iterate through the files which contain information about its
    # 'source'=='category', 'subtype', these are used in UI for document selection
    #  which will be used later for filtering database
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

                # create the chunks
                doc_processed = text_splitter.split_documents(docs[file])
                print("chunks in subtype:",subtype, "are:",len(doc_processed))

                # add metadata information 
                for doc in doc_processed:
                    doc.metadata["source"] = category
                    doc.metadata["subtype"] = subtype
                    doc.metadata["year"] = file[-4:]
                    doc.metadata["filename"] = file

                all_documents[category].append(doc_processed)
    
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
        encode_kwargs = {'normalize_embeddings': True},
        model_name="BAAI/bge-large-en-v1.5"
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

def get_local_qdrant(): 
    qdrant_collections = {}
    embeddings = HuggingFaceEmbeddings(
        model_kwargs = {'device': device},
        encode_kwargs = {'normalize_embeddings': True},
        model_name="BAAI/bge-en-icl")
    list_ = ['Consolidated','District','Ministry','allreports']
    for val in list_:
        client = QdrantClient(path=f"./data/{val}") 
        print(client.get_collections())
        qdrant_collections[val] = Qdrant(client=client, collection_name=val, embeddings=embeddings, )
    return qdrant_collections
    