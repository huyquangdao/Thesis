class BaseParser(object):

    def __init__(self):
        pass

    def parser(self, sent):
        raise NotImplementedError('You must implement this method')


class SpacyParser(BaseParser):

    def __init__(self):
        super(SpacyParser, self).__init__()

    def parser(self, sent):
        pass
