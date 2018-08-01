from ..core import Hermes


class BeliefSlot(object):
    """
    A slot with probability distribution over a finite state of values.
    Mainly used for Speech Input

    Example:
    ```
    slot = BeliefSlot("adjust_value", ["more", "less"], False)
    ```
    Attrbutes:
        name (str): name of the slot
        possible_values (list): default values if provided
        value_conf_map (dict): entity_value -> conf mapping
        last_update_turn_id (int): the last turn this slot was modified
        permit_new (bool): whether this BeliefSlot permits new slots
        validator (function): a function to validate the values
    """

    def __init__(self, name, threshold=0.8, possible_values=[], permit_new=True, validator=None):
        """
        Args:
            name (str): slot name
            threshold (float): confidence threshold
            values (list): value names used as default
            permit_new (bool): whether an unseen value in the possible_value list can be added
            validator (method): a method that validates the values
        """
        self.name = name
        self.threshold = threshold
        self.possible_values = possible_values
        self.permit_new = permit_new
        self.validator = validator

        # Reset slot
        self.reset()

    def reset(self):
        """
        Resets last_update_turn_id and value_conf_map
        """
        self.last_update_turn_id = 0
        self.value_conf_map = {v: 0. for v in self.possible_values}

    def add_observation(self, value, conf, turn_id):
        """
        Args:
            value (object):
            conf  (float):
            turn_id (int):
        Returns:
            bool: True is successful, False otherwise
        """
        # validate args
        if self.validator is not None and not self.validator(value):
            return False
        if not self.permit_new and value not in self.value_conf_map:
            return False
        if not isinstance(conf, float) or not 0 <= conf <= 1:
            return False

        # Simply assign confidence for now
        self.value_conf_map[value] = conf
        self.last_update_turn_id = turn_id
        return True

    def get_max_conf_value(self):
        """
        Returns:
            value (str): 
            conf (float):
        """
        if len(self.value_conf_map) == 0:
            return None, -1
        max_value, max_conf = max(
            self.value_conf_map.items(), key=lambda tup: tup[1])
        if max_conf <= 0:  # For neglected stuff
            return None, -1
        return max_value, max_conf

    def get_max_conf(self):
        _, max_conf = self.get_max_conf_value()
        return max_conf

    def get_max_value(self):
        max_value, _ = self.get_max_conf_value()
        return max_value

    def to_json(self):
        """ 
        Returns:
            obj (dict): slot dict
        """
        max_value, max_conf = self.get_max_conf_value()
        if max_value is None:
            return Hermes.build_slot_dict(self.name)
        else:
            return Hermes.build_slot_dict(self.name, max_value, max_conf)


class PSToolSlot(BeliefSlot):
    """
    Used for Photoshop Tool input
    The value of slot that is obtained by interacting directly with Photoshop.
    Therefore, only one value can be present, and has confidence 1.0
    """

    def add_observation(self, value, conf, turn_id):
        """
        Args:
            value (any): value for the slot
            turn_id (int): update turn id
        """
        self.value_conf_map = {v: 0 for v in self.possible_values}
        result = super(PSToolSlot, self).add_observation(
            value, conf, turn_id)
        return result
