from llama_index import VectorStoreIndex, SimpleDirectoryReader
from os import path, environ
from query_eng import *

# api_key
path_to_key = r"C:\Users\User\Desktop\Учёба\опд\траю лламу"
key_name = 'api key.txt'
full_key_path = path.join(path_to_key, key_name)
api_key = open(full_key_path, 'r').readline()
environ['OPENAI_API_KEY'] = api_key

#
answering = query_eng()
answering.query_eng()
