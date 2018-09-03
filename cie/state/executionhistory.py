class ExecutionHistory(object):
    """
    A C++ like stack to store current intent slots
    Also provides search functions like visionengine
    """

    def __init__(self):
        self._stack = list()

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

    def empty(self):
        return self.size() == 0

    def clear(self):
        return self._stack.clear()

    def select_object(self, object, **kwargs):
        """
        Args
            object (str): name of object
        Returns:
            mask_strs (list): list of mask strings
        """
        if object == "image":
            return []

        mask_strs = []
        for intent_tree in self._stack:
            node_dict = intent_tree.node_dict
            if object == node_dict['object'].get_max_value():
                object_mask_str = node_dict['object_mask_str'].get_max_value()
                mask_strs.append(object_mask_str)

        # Remove duplicates
        mask_strs = list(set(mask_strs))
        return mask_strs
