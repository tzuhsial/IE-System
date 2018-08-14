from collections import OrderedDict
import copy
import logging
import sys

from ..core import SysIntent
from ..util import build_slot_dict, find_slot_with_key, slots_to_args

logger = logging.getLogger(__name__)


class BeliefNode(object):
    """
    A slot node with probability distribution over some values.
    When pulled, addes the intent of current node
    Example:
    ```
    adjust_value_node = BeliefNode("adjust_value", ["more", "less"], False)
    ```
    Attrbutes:
        name (str): name of the slot
        possible_values (list): list of valid values for this particular node
        value_conf_map (dict): entity_value -> conf mapping
        last_update_turn_id (int): the last turn this slot was modified
        permit_new (bool): whether this BeliefSlot permits new slots
        validator (function): a function to validate the values
    """

    def __init__(self, name, threshold=0.7, possible_values=None, validator=None, **kwargs):
        """
        Args:
            name (str): slot name
            threshold (float): confidence threshold
            values (list): possible values
            validator (method): a method that validates observed values
        """
        self.name = name
        self.threshold = threshold
        self.possible_values = [] if possible_values is None else possible_values
        self.validator = validator

        # Reset slot
        self.last_update_turn_id = 0
        self.value_conf_map = OrderedDict(
            {v: 0. for v in self.possible_values})

        # Dependency Graph Related
        self.children = {}
        self.optional = {}

        # System Intents
        self.intent = SysIntent()

    def clear(self):
        """
        Reset last_update_turn_id, value_conf_map & intent
        """
        self.last_update_turn_id = 0
        self.value_conf_map = {v: 0. for v in self.possible_values}
        self.intent.clear()

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
            logger.debug("invalid value: {}".format(value))
            return False
        if len(self.possible_values) and value not in self.value_conf_map:
            return False
        if (not isinstance(conf, float) and not isinstance(conf, int)) or \
                (not 0 <= conf <= 1):
            logger.debug("invalid confidence: {}".format(conf))
            return False

        # A new value observed indicates that previous value should have lower confidence
        for key in self.value_conf_map:
            self.value_conf_map[key] -= 0.1
            logger.debug("node {} decayed value {} conf to {}"
                         .format(self.name, key, self.value_conf_map[key]))

        # Simply assign confidence for now
        self.value_conf_map[value] = conf
        self.last_update_turn_id = turn_id
        return True

    def get_max_conf_value(self):
        """
        Returns the value & confidence pair with max confidence
        Returns:
            value (obj):
            conf (float):
        """
        if len(self.value_conf_map) == 0:
            return None, 0.
        max_value, max_conf = max(
            self.value_conf_map.items(), key=lambda tup: tup[1])
        if max_conf <= 0:  # For neglected stuff
            return None, 0.
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
            return build_slot_dict(self.name)
        else:
            return build_slot_dict(self.name, max_value, max_conf)

    #########################
    #     Graph Related     #
    #########################
    def add_child(self, node, optional=False):
        """
        adds children 
        Args:
            node (obj): the child node to be added
            optional (bool): whether this child is optional
        """
        if node.name not in self.children:
            self.children[node.name] = node
            self.optional[node.name] = optional
            return True
        else:
            logger.debug("node {} is already a child of node {}!".format(
                node.name, self.name))
            return False

    def pull(self):
        """
        Note: Customized nodes should override this function
        Returns the intent of the node as tree.

        Update the intent of the current node
        Returns:
            self.intent (object)
        """

        self.intent = self._build_slot_intent()

        # Leaf node
        if len(self.children) == 0:
            return self.intent

        # Else not leaf, build intent from children
        children_intent, update_turn_id = self._build_children_intent()

        # Current node does not need to be updated
        if not children_intent.empty():
            self.last_update_turn_id = update_turn_id
            self.intent += children_intent

        return self.intent

    def _build_slot_intent(self):
        """
        Construct intent with the value of the slot node
        """
        intent = SysIntent()
        slot = self.to_json()
        max_conf = slot.get('conf', 0.0)
        if max_conf < 0.5:
            intent.request_slots.append(slot)
        elif max_conf < self.threshold:
            intent.confirm_slots.append(slot)
        else:
            intent.execute_slots.append(slot)
        return intent

    def _build_children_intent(self):
        """
        Returns:
            children_intent (obj): the combined intent of children nodes
            updated_turn_id (int): the latest turn id needs to be updated
        """
        # Initialize return arguments
        children_intent = SysIntent()
        update_turn_id = self.last_update_turn_id

        # Loop through children
        for child_name, child in self.children.items():
            if child.last_update_turn_id >= self.last_update_turn_id:
                child_intent = child.pull()
                optional = self.optional[child_name]
                if optional:
                    child_intent.request_slots = []  # Optional child node's request slots is empty!
                children_intent += child_intent

            update_turn_id = max(child.last_update_turn_id, update_turn_id)

        return children_intent, update_turn_id


class IntentNode(BeliefNode):
    """
    Root of a intent tree
    Does not store any sysintent values, only from children
    Also provides convenient access to every slot node in the tree
    Attributes:
        name (str)
        children (dict)
        optional (dict)
        node_dict (dict)
        last_update_turn_id (int)
        intent (SysIntent)
    """

    def __init__(self, name, **kwargs):
        self.name = name

        self.children = {}
        self.optional = {}

        self.node_dict = {}

        self.last_update_turn_id = -1

        self.intent = SysIntent()

    def pull(self):
        """
        Always pull from children, so set last_update_turn_id to -1
        """
        sysintent = super(IntentNode, self).pull()
        self.last_update_turn_id = -1
        return sysintent

    def add_observation(self):
        raise NotImplementedError

    def build_node_dict(self):
        """
        Clear the values & intent of the tree
        """
        self.node_dict = {}
        queue = []
        queue.append(self)
        while len(queue):
            node = queue.pop(0)
            self.node_dict[node.name] = node
            queue += list(node.children.values())

    def get_slot(self, slot_name):
        return self.node_dict[slot_name]

    def clear(self):
        """
        Clear the values & intent of itself and all the children
        """
        self.intent.clear()
        queue = []
        queue += list(self.children.values())
        while len(queue):
            node = queue.pop(0)
            node.clear()
            queue += list(node.children.values())

    def get_max_conf_value(self):
        return self.name, 1.0

    def to_json(self):
        return build_slot_dict('intent', self.name, 1.0)

    def _build_slot_intent(self):
        """
        IntentNode does not have value of its own
        """
        return SysIntent()

    def _build_children_intent(self):
        """
        For Root Node, update all children intent, regardless of update_turn_id

        Returns:
            children_intent (obj): the combined intent of children nodes
            updated_turn_id (int): the latest turn id needs to be updated
        """
        # Initialize return arguments
        children_intent = SysIntent()
        update_turn_id = self.last_update_turn_id

        # Loop through children
        for child_name, child in self.children.items():
            child_intent = child.pull()
            optional = self.optional[child_name]
            if optional:
                child_intent.request_slots = []  # Optional child node's request slots is empty!
            children_intent += child_intent

        update_turn_id = max(child.last_update_turn_id, update_turn_id)

        return children_intent, update_turn_id


class PSToolNode(BeliefNode):
    """
    A node that allows only one value with 100% confidence
    """

    def __init__(self, name, threshold=1.0, possible_values=None, validator=None, **kwargs):
        """
        Threshold should always be 1.0, and no restriction on possible values
        """
        assert threshold == 1.0
        assert possible_values == None
        super(PSToolNode, self).__init__(
            name, threshold, possible_values, validator)

    def add_observation(self, value, conf, turn_id):
        """
        Only one value can be present at the time
        """
        if conf < 1.0:
            logger.error(
                "PSToolNode {} observed confidence less than 1.0!".format(self.name))
            return False
        prev_value_conf_map = copy.deepcopy(self.value_conf_map)
        self.value_conf_map = {}
        result = super(PSToolNode, self).add_observation(value, conf, turn_id)
        if not result:
            self.value_conf_map = prev_value_conf_map
        return True


class ObjectMaskStrNode(BeliefNode):
    """
    Customized node for object_mask_str
    There are some hard coded stuff though
    object_mask_str has the ability to query cv engine

    There are 3 child nodes
    1. b64_img_str
    2. object
    3. object_mask_id

    Attrbutes:
        name (str): name of the slot
        possible_values (list): list of valid values for this particular node
        value_conf_map (dict): entity_value -> conf mapping
        last_update_turn_id (int): the last turn this slot was modified
        permit_new (bool): whether this BeliefSlot permits new slots
        validator (function): a function to validate the values
        visionengine (object): shared visionengine object to perform select_object queries
    """

    def __init__(self, name="object_mask_str", threshold=0.8, possible_values=None, validator=None, visionengine=None, **kwargs):
        assert name == "object_mask_str", "name should be ObjectMaskStrNode"
        super(ObjectMaskStrNode, self).__init__(
            name, threshold, possible_values, validator)
        if visionengine is None:
            logger.warning("ObjectMaskStrNode.visionengine is not loaded!")
        self.visionengine = visionengine

    def query(self):
        b64_img_str = self.children["b64_img_str"].get_max_value()
        object_intent = self.children['object'].intent
        args = slots_to_args(object_intent.execute_slots)
        args['b64_img_str'] = b64_img_str

        if args.get("object") is None or args.get('b64_img_str') is None:
            logger.error("Missing object or b64_img_str")
            mask_strs = []
        else:
            mask_strs = self.visionengine.select_object(**args)
        return mask_strs

    def pull(self):
        """
        ObjectMaskNode should have 3 children
        1. object
        2. b64_img_str
        3. object_mask_id
        Returns:
            self.intent (object)
        """

        assert sorted(self.children.keys()) == [
            'b64_img_str', 'object', 'object_mask_id'], "ObjectMaskNode has wrong dependencies"

        #b64_img_str_node = self.children['b64_img_str']
        object_node = self.children['object']
        object_mask_id_node = self.children['object_mask_id']

        # Since object_mask is internal,
        # we first update its turn_id to the latest according to its child nodes

        # Update current last_update_turn_id to the newest
        object_mask_str_turn_id = self.last_update_turn_id
        object_turn_id = object_node.last_update_turn_id
        object_mask_id_turn_id = object_mask_id_node.last_update_turn_id

        self.last_update_turn_id = max(
            object_mask_str_turn_id, object_turn_id, object_mask_id_turn_id)

        # Special case, where both child nodes are updated
        if object_turn_id >= self.last_update_turn_id and \
           object_mask_id_turn_id >= self.last_update_turn_id:
            logger.warning(
                "object and object_mask_id are both updated at the same turn!")
            # In fact, this case is handled by forcing object_mask_id_slot
            # to be cleared when we query the cv engine

        # Pull from object_node
        object_intent = object_node.pull()
        if object_turn_id >= self.last_update_turn_id:
            # Pull from object_intent
            if not object_intent.executable():
                self.intent = object_intent
                return self.intent

            if object_node.get_max_value() == "image":  # Special case: image
                self.intent = SysIntent()
                return self.intent

            # Query CV engine: a lazy update
            object_mask_id_node.clear()  # This prevents pulling from object_mask_id

            mask_strs = self.query()
            for mask_str in mask_strs:
                # Directly modify value_conf_map
                # Since add_observation will decay previous observations
                self.value_conf_map = {
                    mask_str: 0.5 for mask_str in mask_strs}

        # Pull from object_mask_id_node
        object_mask_id_intent = object_mask_id_node.pull()
        object_mask_id_turn_id = object_mask_id_node.last_update_turn_id
        if object_mask_id_turn_id >= self.last_update_turn_id:

            # object_mask_id
            if not object_mask_id_intent.executable():
                self.intent = object_mask_id_intent
                return self.intent

            object_mask_id = object_mask_id_node.get_max_value()

            if object_mask_id == -1:
                for value in self.value_conf_map:
                    self.value_conf_map[ value ] = 0.0 
            elif object_mask_id < len(self.value_conf_map):
                candidates = list(self.value_conf_map.items())
                mask_str, _ = candidates[object_mask_id]

                self.value_conf_map.clear()
                self.add_observation(mask_str, 1.0, self.last_update_turn_id)
            else:
                object_mask_id_node.clear()

        # Building self intent, which includes requesting object_mask_id
        self.intent = self._build_slot_intent()
        return self.intent

    def _build_slot_intent(self):
        """
        Build intent depending on the values
        The intent after obtaining mask_str from the vision engine
        There should only be 3 types: 1.request 2 confirm 3. execute
        """
        intent = SysIntent()

        if self.get_max_conf() >= self.threshold:
            mask_str, conf = self.get_max_conf_value()
            mask_str_slot = build_slot_dict('object_mask_str', mask_str, conf)
            intent.execute_slots.append(mask_str_slot)
            return intent

        if self.get_max_conf() < 0.5:
            # Request object_mask_str directly
            mask_str_slot = build_slot_dict('object_mask_str')
            intent.request_slots.append(mask_str_slot)
            return intent

        # Filter out values that are between threshold and 0.5
        mask_strs = list(self.value_conf_map.keys())

        if len(mask_strs) == 1:  # Confirm, since there is only 1 result
            mask_str = mask_strs[0]
            mask_str_slot = build_slot_dict('object_mask_str', mask_str)
            intent.confirm_slots.append(mask_str_slot)
        else:  # > 1, request object_mask_id
            object_mask_id_slot = build_slot_dict('object_mask_id', mask_strs)
            intent.request_slots.append(object_mask_id_slot)

        return intent


def builder(string):
    """
    Gets node class with string
    """
    try:
        return getattr(sys.modules[__name__], string)
    except AttributeError:
        logger.error("Unknown node: {}".format(string))
        return None
