import re
from scripts.dataset.module.corenlp import Corenlp
from scripts.dataset.pre_process.models import Segmenter


class CorenlpSegmenter(Segmenter):
    def __init__(self):
        super().__init__()

    def segment(self, text):
        """
        :param string text: document that needs to be segmented
        :return: list of string
        """
        # remove (ABSTRACT TRUNCATED AT 250 WORDS)
        text = re.sub('\(ABSTRACT TRUNCATED AT 250 WORDS\)', '', text)

        ann = Corenlp.annotate(text, annotators=[Corenlp.SEGMENTER])
        ret = [text[s.characterOffsetBegin:s.characterOffsetEnd] for s in ann.sentence]

        return ret
