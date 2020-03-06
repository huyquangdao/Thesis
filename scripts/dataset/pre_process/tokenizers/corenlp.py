import re
from scripts.dataset.module.corenlp import Corenlp
from scripts.dataset.pre_process.models import Tokenizer


class CorenlpTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, sent):
        sent = re.sub(r'\s{2,}', ' ', sent)

        ann = Corenlp.annotate(sent, annotators=[Corenlp.POS, Corenlp.TOKENIZER])

        ret = []
        for s in ann.sentence:
            ret.extend([(sent[t.beginChar:t.endChar], t.pos) for t in s.token])

        return ret
