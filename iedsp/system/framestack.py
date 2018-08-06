
class FrameStack(object):
    """
    A C++ like stack to store intent slots
    Also provides some convenient functions e.g. prioritize
    """

    def __init__(self):
        self._stack = list()

    def __contains__(self, intent_tree):
        return intent_tree in self._stack

    def top(self):
        if len(self._stack):
            return self._stack[0]
        return None

    def push(self, intent_tree):
        self._stack.insert(0, intent_tree)

    def pop(self):
        self._stack.pop(0)

    def size(self):
        return len(self._stack)
