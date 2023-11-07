from llama_index import VectorStoreIndex
from file_reader import *
import os


class query_eng:
    def import_files(self):
        self.f_r_ex = file_reader()
        self.f_r_ex.txt_import()
        if self.f_r_ex.files_with_extension():
            self.f_r_ex.word_2_txt()
        self.docs = self.f_r_ex.return_docs()
        self.index = VectorStoreIndex.from_documents(self.docs)
        return self.index

    def query_eng(self):
        query_engine = self.index.as_query_engine()
        request = input('введите запрос: ')
        response = query_engine.query(request)
        respone_to_print = response.response.split('. ')
        for string in respone_to_print:
            print(string)
