from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index import (ServiceContext, VectorStoreIndex,
                         load_index_from_storage, StorageContext,
                         )
import os
import file_reader
from time import sleep

#
class HF:
    def status(self):
        return len(os.listdir(self.save_path)) > 0

    def __init__(self):
        # api_key
        path_to_key = r"C:\Users\User\Desktop\Учёба\опд\траю лламу"
        key_name = 'api key.txt'
        full_key_path = os.path.join(path_to_key, key_name)
        api_key = open(full_key_path, 'r').readline()
        os.environ['OPENAI_API_KEY'] = api_key
        self.documents = file_reader.file_reader().return_docs()
        self.save_path = r"C:\Users\User\Desktop\Учёба\опд\траю лламу\index data"
        if not self.status():
            embed_model = HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L12-v2')
            service_context = ServiceContext.from_defaults(embed_model=embed_model)
            self.index = VectorStoreIndex.from_documents(documents=
                                                         self.documents, service_context=service_context
                                                         )
            self.index.storage_context.persist(persist_dir=self.save_path)
        else:
            self.storage_context = StorageContext.from_defaults(persist_dir=self.save_path)
            self.index = load_index_from_storage(self.storage_context)
    def build_vectors(self):  # построение и сохранение векторного пространства (только при наличии доков)
        embed_model = HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L12-v2')
        service_context = ServiceContext.from_defaults(embed_model=embed_model)
        self.index = VectorStoreIndex.from_documents(documents=
                                                     self.documents, service_context=service_context
                                                     )
        self.index.storage_context.persist(persist_dir=self.save_path)

    def load_vectors(self):  # загрузка пространства
        self.storage_context = StorageContext.from_defaults(persist_dir=self.save_path)
        self.index = load_index_from_storage(self.storage_context)

    def test(self, request):  # тест пространства
        query_engine = self.index.as_query_engine()
        response = query_engine.query(request)
        print(response)

    def interface(self):
        flag = False
        while not flag:
            print('-' * 18)
            print('интерфейс работы с векторами')
            print('-' * 18)
            print('1 - построить пространство')
            print('2 - загрузить пространство')
            print('3 - произвести тестовый запрос')
            print('4 - выход')
            print('-' * 18)
            request = input('введите необходимую операцию: ')
            print('-' * 18)
            if request == '1':
                self.build_vectors()
                print('-' * 18)
                sleep(2)
            if request == '2':
                self.load_vectors()
                print('-' * 18)
                sleep(2)
            if request == '3':
                req = input('введите запрос: ')
                print('-' * 18)
                try:
                    self.test(req)
                except:
                    print('сначала загрузите пространство')
                sleep(2)

            if request == '4':
                quit()


HF().interface()
