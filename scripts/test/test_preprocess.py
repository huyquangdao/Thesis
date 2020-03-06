from unittest import TestCase
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))
from scripts.dataset.data_readers import CDRReader
from scripts.dataset.pre_process import opt as pre_opt
from scripts.dataset.data_managers import CDRDataManager
from scripts.models import constants
from scripts.dataset import pre_process
import pickle


class TestPreprocess(TestCase):

    def test_preprocess(self):
        file_name = '../../data/cdr_data/cdr_train.txt'
        pre_config = {
            pre_opt.SEGMENTER_KEY: pre_opt.SpacySegmenter(),
            pre_opt.TOKENIZER_KEY: pre_opt.SpacyTokenizer()
        }
        reader = CDRReader(file_name=file_name)
        data_manager = CDRDataManager()
        raw_documents = reader.read()

        title_docs, abstract_docs = data_manager.parse_documents(raw_documents)

        # Pre-process
        title_doc_objs = pre_process.process(title_docs, pre_config, constants.SENTENCE_TYPE_TITLE)
        abs_doc_objs = pre_process.process(abstract_docs, pre_config, constants.SENTENCE_TYPE_ABSTRACT)

        self.assertEqual(len(title_doc_objs), len(abs_doc_objs))

        documents = data_manager.merge_documents(title_doc_objs, abs_doc_objs)

        outfile = '../../data/cdr_data/documents.pkl'

        with open(outfile,'wb') as f:

            pickle.dump(documents,f)