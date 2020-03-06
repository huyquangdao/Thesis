from unittest import TestCase
import sys
from pathlib import Path
import pickle
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))
from scripts.dataset.data_processor import CDRProcessor


class TestCDRProcessor(TestCase):
    def test_generate_data(self):
        file_path = '../../data/cdr_data/cdr_train.txt'
        processor = CDRProcessor(file_path=file_path)
        data_tree = processor.generate_data()
        print(data_tree)
