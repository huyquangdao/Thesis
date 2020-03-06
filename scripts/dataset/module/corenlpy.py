import corenlp
import atexit
from scripts.utils.utils import Timer

class Corenlp:
    RESTART_SERVER_AFTER = 500

    TOKENIZER = 'tokenize'
    SEGMENTER = 'ssplit'
    POS = 'pos'
    DEPPARSE = 'depparse'
    LEMMA = 'lemma'

    client = None
    counter = 0

    DEFAULT_ANNOTATORS = [TOKENIZER, SEGMENTER]
    DEFAULT_PROPERTIES = {}

    @staticmethod
    def annotate(text, annotators=DEFAULT_ANNOTATORS):
        """

        :param str text:
        :param list of str annotators:
        :return:
        """
        if Corenlp.counter == Corenlp.RESTART_SERVER_AFTER:
            # restart server after RESTART_SERVER_AFTER requests
            Corenlp.stop()
            Corenlp.counter = 0

        Corenlp.counter += 1

        if Corenlp.client is None:
            Corenlp.start()

        annotators = set(annotators + Corenlp.DEFAULT_ANNOTATORS)
        properties = Corenlp.DEFAULT_PROPERTIES
        properties.update({
            'annotators': ','.join(annotators),
            'inputFormat': 'text',
            'outputFormat': 'serialized',
            'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer',
            'tokenize.options': 'splitHyphenated=true'
        })
        return Corenlp.client.annotate(text, properties=properties)

    @staticmethod
    def start():
        timer = Timer()
        timer.start('Start CoreNLP server')

        Corenlp.client = corenlp.CoreNLPClient(
            annotators=Corenlp.DEFAULT_ANNOTATORS,
            properties=Corenlp.DEFAULT_PROPERTIES
        )

        timer.stop()

    @staticmethod
    def stop():
        timer = Timer()
        timer.start('Stop CoreNLP server')

        if Corenlp.client is not None:
            Corenlp.client.stop()
            Corenlp.client = None

        timer.stop()


atexit.register(Corenlp.stop)
