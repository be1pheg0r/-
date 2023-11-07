from llama_index import SimpleDirectoryReader
from pdf2docx import parse
import docx2txt
import os


class file_reader:

    def __init__(self):

        self.get_path = r"C:\Users\User\Desktop\Учёба\опд\траю лламу\inp_data"
        self.take_path = r"C:\Users\User\Desktop\Учёба\опд\траю лламу\data"
        self.read_data = None

    def txt_import(self):
        count = False
        for file in os.listdir(self.get_path):
            if os.path.isfile(os.path.join(self.get_path, file)) and file.endswith('.txt'):
                source_file = os.path.join(self.get_path, file)
                next_file = os.path.join(self.take_path, file)
                os.rename(source_file, next_file)
                count = True
        if count:
            print('импорт .txt завершен')

    def files_with_extension(self):
        for file in os.listdir(self.get_path):
            if os.path.isfile(os.path.join(self.get_path, file)):
                if file.endswith('.docx') or file.endswith('.pdf'):
                    return True
        return False

    def return_docs(self):
        if self.files_with_extension():
            self.word_2_txt()
        self.read_data = SimpleDirectoryReader('data').load_data()
        return self.read_data

    # def convert_pdf_to_docx(self):
    #    for root, dirs, files in os.walk(self.get_path):
    #        for file in files:
    #            if file.endswith(".pdf"):
    #                pdf_file_path = os.path.join(root, file)
    #                docx_file_path = os.path.join(self.get_path, os.path.splitext(file)[0] + '.docx')
    #
    #                if not os.path.exists(docx_file_path):
    #                    try:
    #                        parse(pdf_file_path, docx_file_path)
    #                        print(f"Файл {file} успешно преобразован в формат DOCX: {docx_file_path}")
    #                        os.remove(pdf_file_path)@
    #                        print(f"Исходный файл {file} удален.")
    #                    except Exception as e:
    #                        print(f"Ошибка при конвертации файла {file}: {e}")
    #                else:
    #                    print(f"Файл {file} уже отформатирован в формат DOCX: {docx_file_path}")

    def word_2_txt(self):
        for root, dirs, files in os.walk(self.get_path):
            for file in files:
                if file.endswith('.docx') or file.endswith('pdf'):

                    docx_file_path = os.path.join(root, file)

                    text_file_path = os.path.join(self.take_path, os.path.splitext(file)[0] + '.txt')
                    if not os.path.exists(text_file_path):
                        try:
                            text = docx2txt.process(docx_file_path)
                            with open(text_file_path, 'w', encoding='utf-8') as text_file:
                                text_file.write(text)
                            print(f'{file} преобразован успешно')
                            os.remove(docx_file_path)
                        except Exception as e:
                            print(f'ошибка, {file} - {e}')
                    else:
                        print(f'{file} уже отформатирован')
        print('конвертация .docx завершена')


