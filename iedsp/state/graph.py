

class GraphNode(object):
    """
    A basic node used for slot dependency management tree
    """

    def __init__(self):
        pass

    def needs_request(self):
        raise NotImplementedError

    def needs_confirm(self):
        raise NotImplementedError

    def needs_query(self):
        raise NotImplementedError
