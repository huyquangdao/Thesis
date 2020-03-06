from unittest import TestCase
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))
from scripts.dataset.data_readers import CDRReader


class TestCDRReader(TestCase):
    def test_read_dataset(self):
        file_path = '../../data/cdr_data/cdr_train.txt'
        reader = CDRReader(file_name=file_path)
        raw_documents = reader.read()
        raw_entities = reader.read_entity()
        raw_relations = reader.read_relation()
        self.assertEqual(len(raw_documents),500)
        print(raw_relations.keys())
