from collections import OrderedDict
import copy
import logging
import sys

from ..core import SysIntent
from .. import util

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

    LAMBDA = 1.0
    K = 5

    def __init__(self,
                 name,
                 threshold=0.8,
                 possible_values=None,
                 validator=None,
                 **kwargs):
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
            {v: 0.
             for v in self.possible_values})

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

        # Rule-based belief update
        for key in self.value_conf_map:
            self.value_conf_map[key] *= (1 - self.LAMBDA)
            logger.debug("node {} decayed value {} conf to {}".format(
                self.name, key, self.value_conf_map[key]))

        # Simply assign confidence for now
        prev_decayed_conf = self.value_conf_map.get(value, 0.0)
        self.value_conf_map[value] = prev_decayed_conf + self.LAMBDA * conf
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
            return util.build_slot_dict(self.name)
        else:
            return util.build_slot_dict(self.name, max_value, max_conf)

    def to_list(self):
        """
        Convert the status of the slot node to list
        Returns top k sorted values
        If has possible values, return list of conf corresponding to each value => imitates prediction
        Else: return the number of entries in the map & maximum confidence score
        """
        l = []
        if len(self.possible_values) > 0:
            # has possible values
            l = []
            sorted_conf = sorted(
                self.value_conf_map.values(), reverse=True)  # Values
            sorted_topk_conf = sorted_conf[:self.K]
            l = sorted_topk_conf
        else:
            # In this case, it is usually PSToolNode
            # Return number of values in map & max confidence score
            max_conf = self.get_max_conf()
            l = [max_conf]
        return l

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
                if optional:  # Do not request optional children
                    child_intent.request_slots = []
                children_intent += child_intent

            update_turn_id = max(child.last_update_turn_id, update_turn_id)

        return children_intent, update_turn_id


class IntentBeliefNode(BeliefNode):
    """
    While most belief nodes need only top-k sorted values,
    we cannot sort intent, since it will decide what actions to execute
    """

    def to_list(self):
        """
        """
        l = []
        if len(self.possible_values) > 0:
            # Return confidence in sorted order
            l = []
            for value in self.possible_values:
                conf = self.value_conf_map.get(value, 0.0)
                l.append(conf)
        else:
            # In this case, it is usually PSToolNode
            # Return number of values in map & max confidence score
            max_conf = self.get_max_conf()
            l = [max_conf]
        return l


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
        return util.build_slot_dict('intent', self.name, 1.0)

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
                child_intent.request_slots = [
                ]  # Optional child node's request slots is empty!
            children_intent += child_intent

        update_turn_id = max(child.last_update_turn_id, update_turn_id)

        return children_intent, update_turn_id


class PSToolNode(BeliefNode):
    """
    A node that allows only one value with 100% confidence
    """
    LAMBDA = 1.0

    def __init__(self,
                 name,
                 threshold=1.0,
                 possible_values=None,
                 validator=None,
                 **kwargs):
        """
        Threshold should always be 1.0, and no restriction on possible values
        """
        assert threshold == 1.0
        assert possible_values == None
        super(PSToolNode, self).__init__(name, threshold, possible_values,
                                         validator)

    def add_observation(self, value, conf, turn_id):
        """
        Only one value can be present at the time
        """
        class_name = self.__class__.__name__

        if conf < 1.0:
            logger.error("{} {} observed confidence less than 1.0!"\
                .format(class_name, self.name))
            return False
        if value == "":
            logger.info("{} {} observed value empty".format(
                class_name, self.name))
            return False
        prev_value_conf_map = copy.deepcopy(self.value_conf_map)
        self.value_conf_map = {}
        result = super(PSToolNode, self).add_observation(value, conf, turn_id)
        if not result:
            self.value_conf_map = prev_value_conf_map
        return result


class PSInfoNode(PSToolNode):
    """
    Stores information of Photoshop, has same behavior as PSToolNode, 
    but we need to differentiate for state feature representation construction
    """
    pass


class PSBinaryInfoNode(PSToolNode):
    """
    While the confidence is always 1.0, 
    Stores True/False
    Examples:
        has_next_history
        has_previous_history
    """

    def to_list(self):
        """
        Return 1.0 for true and 0.0 for false
        """
        if len(self.value_conf_map) == 0:
            return [0.0]

        assert len(self.value_conf_map) == 1
        value = list(self.value_conf_map.keys())[0]
        assert value is True or value is False
        l = [1.0] if value else [0.0]
        return l


class ObjectMaskStrNode(BeliefNode):
    """
    Customized node for object_mask_str
    There are some hard coded stuff though
    object_mask_str has the ability to query cv engine

    There are 3 child nodes
    1. b64_img_str
    2. object
    3. gesture_click

    Attrbutes:
        name (str): name of the slot
        possible_values (list): list of valid values for this particular node
        value_conf_map (dict): entity_value -> conf mapping
        last_update_turn_id (int): the last turn this slot was modified
        permit_new (bool): whether this BeliefSlot permits new slots
        validator (function): a function to validate the values
    """

    LAMBDA = 1.0

    def __init__(self,
                 name="object_mask_str",
                 threshold=0.8,
                 possible_values=None,
                 validator=None,
                 **kwargs):
        assert name == "object_mask_str", "name should be ObjectMaskStrNode"
        super(ObjectMaskStrNode, self).__init__(name, threshold,
                                                possible_values, validator)

    def pull(self):
        """
        ObjectMaskNode should have 3 children
        1. b64_img_str
        2. object
        3. gesture_click

        Returns:
            self.intent (object)
        """
        assert sorted(self.children.keys()) == ['gesture_click', 'object', 'original_b64_img_str'],\
            "ObjectMaskNode has wrong children"

        object_node = self.children['object']
        gesture_click_node = self.children["gesture_click"]

        # Update current last_update_turn_id according to children
        object_mask_str_turn_id = self.last_update_turn_id
        object_turn_id = object_node.last_update_turn_id
        gesture_click_turn_id = gesture_click_node.last_update_turn_id

        self.last_update_turn_id = max(object_mask_str_turn_id, object_turn_id,
                                       gesture_click_turn_id)

        # Start to do pulling
        self.intent = SysIntent()

        object_updated = object_turn_id >= self.last_update_turn_id
        gesture_updated = gesture_click_turn_id >= self.last_update_turn_id

        if object_updated and gesture_updated:

            object_intent = object_node.pull()
            if not object_intent.executable():
                self.intent = object_intent
                return self.intent

            if object_node.get_max_value() == "image":  # Special case: image
                self.intent = SysIntent()
                return self.intent

            slots = object_intent.execute_slots

            # Query
            gesture_intent = gesture_click_node.pull()
            assert gesture_intent.executable()

            slots += gesture_intent.execute_slots

            self.intent = SysIntent(query_slots=slots)
            return self.intent

        elif object_updated:

            object_intent = object_node.pull()
            if not object_intent.executable():
                self.intent = object_intent
                return self.intent

            if object_node.get_max_value() == "image":  # Special case: image
                self.intent = SysIntent()
                return self.intent

            slots = object_intent.execute_slots
            self.intent = SysIntent(query_slots=slots)
            return self.intent

        elif gesture_updated:
            """
            Case 1: 
                no object, no candidates
            Case 2: 

            """
            gesture_click_intent = gesture_click_node.pull()
            assert gesture_click_intent.executable()

            # Propagate
            gesture_click = gesture_click_node.get_max_value()
            self.filter_candidates(gesture_click)

        self.intent = self._build_slot_intent()
        return self.intent

    def filter_candidates(self, gesture_click):
        """
        Filter candidates in value_conf_map with gesture_click
        Returns:
            True if gesture_click if sure, else False
        """
        n_clicked = 0

        for mask_candidate in self.value_conf_map:
            n_clicked += self.check_overlap(gesture_click, mask_candidate)
            self.value_conf_map[mask_candidate] = 0.7

        if n_clicked == 0 or n_clicked > 1:
            self.value_conf_map.clear()
            return False
        else:
            value, conf = self.get_max_conf_value()
            self.value_conf_map = {value: conf}
        return True

    def _build_slot_intent(self):
        """
        Build intent depending on the values
        The intent after obtaining mask_str from the vision engine
        There should only be 3 types: 1.request 2 confirm 3. execute
        """
        intent = SysIntent()

        if self.get_max_conf() >= self.threshold:
            mask_str_slot = self.to_json()
            intent.execute_slots.append(mask_str_slot)
            return intent

        if self.get_max_conf() < 0.5:
            # Request object_mask_str directly
            mask_str_slot = util.build_slot_dict('object_mask_str')
            intent.request_slots.append(mask_str_slot)
            return intent

        # Filter out values that are between threshold and 0.5
        mask_strs = list(self.value_conf_map.keys())
        if len(mask_strs) == 1:  # Confirm, since there is only 1 result
            mask_str = mask_strs[0]
            mask_str_slot = util.build_slot_dict('object_mask_str', mask_str)
            intent.confirm_slots.append(mask_str_slot)
        else:  # > 1, request object_mask_id
            gesture_click_slot = util.build_slot_dict('gesture_click')
            intent.request_slots.append(gesture_click_slot)

        return intent

    def check_overlap(self, gesture_click, mask_candidate):
        """
        Check if gesture_click(mask) & candidate(mask) overlaps
        """
        gesture_mask = util.b64_to_img(gesture_click)
        candidate_mask = util.b64_to_img(mask_candidate)
        assert gesture_mask.shape == candidate_mask.shape
        return (gesture_mask & candidate_mask).sum() > 0


def builder(string):
    """
    Gets node class with string
    """
    try:
        return getattr(sys.modules[__name__], string)
    except AttributeError:
        logger.error("Unknown node: {}".format(string))
        return None
