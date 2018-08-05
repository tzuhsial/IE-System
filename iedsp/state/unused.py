from .graph import BeliefNode

class PSToolNode(BeliefNode):
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
