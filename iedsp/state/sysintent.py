from ..util import sort_slots_with_key


class SysIntent(object):
    def __init__(self, confirm_slots=None, request_slots=None, label_slots=None, execute_slots=None):
        """
        default argument cannot be list -> a bug that fucked me for 2 days...
        """
        if confirm_slots is None:
            confirm_slots = []
        if request_slots is None:
            request_slots = []
        if label_slots is None:
            label_slots = []
        if execute_slots is None:
            execute_slots = []
        self.confirm_slots = confirm_slots
        self.request_slots = request_slots
        self.label_slots = label_slots
        self.execute_slots = execute_slots

    def __iadd__(self, other_intent):
        self.confirm_slots += other_intent.confirm_slots
        self.request_slots += other_intent.request_slots
        self.label_slots += other_intent.label_slots
        self.execute_slots += other_intent.execute_slots
        return self

    def __add__(self, other_intent):
        confirm_slots = self.confirm_slots + other_intent.confirm_slots
        request_slots = self.request_slots + other_intent.request_slots
        label_slots = self.label_slots + other_intent.label_slots
        execute_slots = self.execute_slots + other_intent.execute_slots
        return SysIntent(confirm_slots, request_slots, label_slots, execute_slots)

    def __radd__(self, other_intent):
        return other_intent.__add__(self)

    def __eq__(self, other_intent):
        """
        Compare slots, regardless of order
        """
        if sort_slots_with_key('slot', self.confirm_slots) != \
           sort_slots_with_key('slot', other_intent.confirm_slots):
            return False
        if sort_slots_with_key('slot', self.request_slots) != \
           sort_slots_with_key('slot', other_intent.request_slots):
            return False
        if sort_slots_with_key('slot', self.label_slots) != \
           sort_slots_with_key('slot', other_intent.label_slots):
            return False
        if sort_slots_with_key('slot', self.execute_slots) != \
           sort_slots_with_key('slot', other_intent.execute_slots):
            return False
        return True

    def empty(self):
        if len(self.confirm_slots) != 0:
            return False
        elif len(self.request_slots) != 0:
            return False
        elif len(self.label_slots) != 0:
            return False
        elif len(self.execute_slots) != 0:
            return False
        return True

    def clear(self):
        self.confirm_slots.clear()
        self.request_slots.clear()
        self.label_slots.clear()
        self.execute_slots.clear()

    def copy(self):
        return SysIntent(self.confirm_slots, self.request_slots, self.label_slots, self.execute_slots)

    def to_json(self):
        obj = {
            'confirm': self.confirm_slots,
            'request': self.request_slots,
            'label': self.label_slots,
            'execute': self.execute_slots
        }
        return obj

    def pprint(self):
        print('confirm', self.confirm_slots)
        print('request', self.request_slots)
        print('label', self.label_slots)
        print('execute', self.execute_slots)

    def executable(self):
        """
        Returns True iff all slots are executable
        """
        if len(self.confirm_slots) != 0:
            return False
        if len(self.request_slots) != 0:
            return False
        if len(self.label_slots) != 0:
            return False
        if len(self.execute_slots) != 0:
            return True
        return False
