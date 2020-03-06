from unittest import TestCase
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))
from scripts.dataset.data_managers import CDRDataManager
from scripts.dataset.data_readers import CDRReader


class TestCDRDataManager(TestCase):

    def test_cdr_data_manager_parse_documents(self):

        file_name = '../../data/cdr_data/cdr_train.txt'
        data_reader = CDRReader(file_name=file_name)
        data_manager = CDRDataManager()

        raw_documents = data_reader.read()

        titles_docs,abstract_doc = data_manager.parse_documents(raw_documents)

        self.assertEqual(len(titles_docs),len(abstract_doc))
        self.assertEqual(len(titles_docs),500)

    def test_cdr_data_manager_merge_documents(self):
        file_name = '../../data/cdr_data/cdr_train.txt'
        data_reader = CDRReader(file_name=file_name)
        data_manager = CDRDataManager()

        raw_documents = data_reader.read()

        title_docs, abstract_docs = data_manager.parse_documents(raw_documents)
        documents = data_manager.merge_documents(title_docs, abstract_docs)
        self.assertEqual(len(documents),500)
