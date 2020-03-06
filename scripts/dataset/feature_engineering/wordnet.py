from itertools import chain

from nltk.corpus import wordnet as wn
from nltk import WordNetLemmatizer

from feature_engineering.word_embedding.nlplab import NLPLab
from scipy.spatial import distance


class WordNet:
    lemmer = None

    @staticmethod
    def get_wordnet_synonyms_nlplab(word, pos=''):
        """
        :param string word: input word
        :param string pos: penn treebank POS tag
        :return:
        """
        syns = WordNet.synonym(word, pos)
        syns = [s for s in syns if s != word and "_" not in s]
        based_vect = NLPLab.word2vec(word)
        syn_vects = [NLPLab.word2vec(w) for w in syns]

        for idx, vec in enumerate(syn_vects):
            if not vec.any():
                syns.pop(idx)
                syn_vects.pop(idx)

        dists = [(syns[idx], distance.euclidean(syn_vect, based_vect)) for idx, syn_vect in enumerate(syn_vects)]
        dists = sorted(dists, key=lambda dist: dist[1])

        if len(dists) != 0:
            return dists[0][0], dists[len(dists) // 2][0]
        else:
            return "$UNK$", "$UNK$"

    @staticmethod
    def get_wordnet_synonyms_fasttext(word, pos=''):
        """
        :param string word: input word
        :param string pos: penn treebank POS tag
        :return:
        """
        ret = ('closest', 'median')
        # Do something here
        return ret

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
        """
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            # As default pos in lemmatization is Noun
            return wn.NOUN

    @staticmethod
    def synonym(word, pos=''):
        """

        :param pos:
        :param word:
        :return:
        """
        synonyms = wn.synsets(word, WordNet.get_wordnet_pos(pos))
        lemmas = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))

        return lemmas

    @staticmethod
    def lemmatize(word, pos):
        if WordNet.lemmer is None:
            WordNet.lemmer = WordNetLemmatizer()

        return WordNet.lemmer.lemmatize(word, WordNet.get_wordnet_pos(pos))


def main():
    # for i in range(1000000):
        print(WordNet.get_wordnet_synonyms_nlplab("hello"))
        # if i % 2000 == 0:
        #     print(i)
