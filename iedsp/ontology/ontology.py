from collections import defaultdict
import logging

from .node import *
from ..visionengine import VisionEnginePortal

logger = logging.getLogger(__name__)


def OntologyPortal(config):
    if config["ONTOLOGY"] == "ImageEditOntology":
        return ImageEditOntology(config)
    else:
        logging.error("Unknown ontology: {}".format(config["ONTOLOGY"]))
        return None


class BaseOntology(object):
    """
    Base Class for Ontology Definitions
    """
    class Intents:
        """
        Define intents for this domain
        """
        UNKNOWN = "unknown"

    class Slots:
        """
        Defines slots for this domain
        """
        INTENT = "intent"

    def __init__(self):
        raise NotImplementedError

    def create(self):
        raise NotImplementedError


class ImageEditOntology(object):
    """
    Ontology for the image edit domain
    """

    class Intents:
        UNKNOWN = "unknown"
        OPEN = "open"
        LOAD = "load"
        CLOSE = "close"
        SELECT_OBJECT = "select_object"
        ADJUST = "adjust"
        UNDO = "undo"
        REDO = "redo"

    class Slots:
        INTENT = "intent"
        IMAGE_PATH = 'image_path'
        B64_IMG_STR = 'b64_img_str'
        ATTRIBUTE = 'attribute'
        ADJUST_VALUE = 'adjust_value'
        OBJECT = 'object'
        OBJECT_MASK = 'object_mask'
        OBJECT_MASK_ID = 'object_mask_id'
        POSITION = "position"
        ADJECTIVE = "adjective"

        # TODO
        COLOR = "color"

    def __init__(self, ontology_config):
        """
        Arguments are possible objects to create a new intent tree
        Args:
            config (dict): ontology config
            visionengine (object): vision engine used in node creation
        """
        self.config = ontology_config
        self.visionengine = None

        self.thresh = self.load_thresh(ontology_config)

    def load_visionengine(self, visionengine):
        self.visionengine = VisionEnginePortal(visionengine)

    def load_thresh(self, ontology_config):
        """
        Loads threshold for slot from config
        Example: set ATTRIBUTE threshold to 0.8 in config.ini
        [ONTOLOGY]
        ATTRIBUTE_THRESHOLD = 0.8
        """
        # Build defualt dict
        if "DEFAULT_THRESHOLD" not in ontology_config:
            logger.warning(
                "DEFAULT_THRESHOLD not defined in [ONTOLOGY] section. Used 0.7")
            default_thresh = 0.7
        else:
            default_thresh = ontology_config["DEFAULT_THRESHOLD"]

        logger.debug("Set DEFAULT_THRESHOLD to {}.".format(default_thresh))

        thresh_dict = defaultdict(lambda: default_thresh)

        # Find all slot names tuple pairs e.g. ("IMAGE_PATH", "image_path")
        slot_name_tuples = filter(
            lambda tup: not tup[0].startswith("_"), vars(self.Slots).items())

        for tup in slot_name_tuples:
            slot_name_upper, slot_name = tup
            config_key = slot_name_upper + "_THRESHOLD"  # IMAGE_PATH_THRESHOLD
            slot_thresh = ontology_config.get(config_key, default_thresh)
            thresh_dict[slot_name] = slot_thresh
            logger.debug("Set \"{}\" threshold {}".format(
                slot_name, slot_thresh))

        return thresh_dict

    def create(self, intent):
        """
        Creates intent tree
        Args:
            intent (str)
        Returns:
            tree (node) : root of the intent tree
            slot_nodes (dict): name of key & node as value
        """
        if self.visionengine is None:
            logger.warning("visionengine not loaded for ontology!")

        # Create intent tree
        if intent == self.Intents.UNKNOWN:
            tree = self.create_unknown()
        elif intent == self.Intents.OPEN:
            tree = self.create_open()
        elif intent == self.Intents.LOAD:
            tree = self.create_load()
        elif intent == self.Intents.SELECT_OBJECT:
            tree = self.create_select_object()
        elif intent == self.Intents.ADJUST:
            tree = self.create_adjust()
        elif intent == self.Intents.UNDO:
            tree = self.create_undo()
        elif intent == self.Intents.REDO:
            tree = self.create_redo()
        elif intent == self.Intents.CLOSE:
            tree = self.create_close()
        else:
            # funny that unknown is actually an intent
            logging.error("Unknown intent: {}!".format(intent))
            tree = None
        return tree

    def create_unknown(self):
        """
        This intent is a special case where the intent is unknown.
        The system will attempt to request or confirm the intent before
        actually stacking the intent tree.

        Dependency tree
        unknown
        |- intent
        """
        unknown_node = IntentNode(self.Intents.UNKNOWN)

        intent_thresh = self.thresh.get(self.Slots.INTENT)
        intent_node = BeliefNode(self.Slots.INTENT, threshold=intent_thresh)

        unknown_node.add_child(intent_node)

        # Build node dict
        unknown_node.build_node_dict()
        return unknown_node

    def create_open(self):
        """
        Dependency tree:
        open
        |- image_path
        """
        open_node = IntentNode(self.Intents.OPEN)
        image_path_node = PSToolNode(self.Slots.IMAGE_PATH)  # threshold is 1.0
        open_node.add_child(image_path_node)

        # Builde node dict
        open_node.build_node_dict()
        return open_node

    def create_load(self):
        """
        Dependency tree:
        load
        |- b64_img_str
        """
        load_node = IntentNode(self.Intents.LOAD)
        b64_img_str_node = PSToolNode(
            self.Slots.B64_IMG_STR)  # threshold is 1.0
        load_node.add_child(b64_img_str_node)

        # Build node dict
        load_node.build_node_dict()
        return load_node

    def create_select_object(self):
        """
        Dependency tree:
        select_object
        |- object_mask
            |- b64_img_str
            |- object
                |- position(optional)
                |- adjective(optional)
            |- object_mask_id
        """
        select_node = IntentNode(self.Intents.SELECT_OBJECT)

        object_mask_node = ObjectMaskNode(
            self.Slots.OBJECT_MASK, visionengine=self.visionengine)

        b64_img_str_node = PSToolNode(self.Slots.B64_IMG_STR)

        object_thresh = self.thresh.get(self.Slots.OBJECT)
        object_node = BeliefNode(
            self.Slots.OBJECT, threshold=object_thresh)

        object_mask_id_thresh = self.thresh.get(self.Slots.OBJECT_MASK_ID)
        object_mask_id_node = BeliefNode(
            self.Slots.OBJECT_MASK_ID, threshold=object_mask_id_thresh)

        position_thresh = self.thresh.get(self.Slots.POSITION)
        position_node = BeliefNode(
            self.Slots.POSITION, threshold=position_thresh)
        adjective_thresh = self.thresh.get(self.Slots.ADJECTIVE)
        adjective_node = BeliefNode(
            self.Slots.ADJECTIVE, threshold=adjective_thresh)

        # Build dependencies
        select_node.add_child(object_mask_node)
        object_mask_node.add_child(b64_img_str_node)
        object_mask_node.add_child(object_node)
        object_mask_node.add_child(object_mask_id_node)

        object_node.add_child(position_node, optional=True)
        object_node.add_child(adjective_node, optional=True)

        # Build node dict
        select_node.build_node_dict()
        return select_node

    def create_adjust(self):
        """
        Dependency tree:
        adjust 
        |- object_mask
            |- b64_img_str
            |- object
                |- position(optional)
                |- adjective(optional)
            |- object_mask_id
        |- attribute
        |- adjust_value
        """
        adjust_node = IntentNode(self.Intents.ADJUST)

        object_mask_node = ObjectMaskNode(
            self.Slots.OBJECT_MASK, visionengine=self.visionengine)
        attribute_thresh = self.thresh.get(self.Slots.ATTRIBUTE)
        attribute_node = BeliefNode(
            self.Slots.ATTRIBUTE, threshold=attribute_thresh)
        adjust_value_thresh = self.thresh.get(self.Slots.ADJUST_VALUE)
        adjust_value_node = BeliefNode(
            self.Slots.ADJUST_VALUE, threshold=adjust_value_thresh)

        b64_img_str_node = PSToolNode(self.Slots.B64_IMG_STR)
        object_thresh = self.thresh.get(self.Slots.OBJECT)
        object_node = BeliefNode(self.Slots.OBJECT, threshold=object_thresh)

        object_mask_id_thresh = self.thresh.get(self.Slots.OBJECT_MASK_ID)
        object_mask_id_node = BeliefNode(
            self.Slots.OBJECT_MASK_ID, threshold=object_mask_id_thresh)

        position_thresh = self.thresh.get(self.Slots.POSITION)
        position_node = BeliefNode(
            self.Slots.POSITION, threshold=position_thresh)
        adjective_thresh = self.thresh.get(self.Slots.ADJECTIVE)
        adjective_node = BeliefNode(
            self.Slots.ADJECTIVE, threshold=adjective_thresh)

        # Build dependencies
        adjust_node.add_child(object_mask_node)
        adjust_node.add_child(attribute_node)
        adjust_node.add_child(adjust_value_node)

        object_mask_node.add_child(b64_img_str_node)
        object_mask_node.add_child(object_node)
        object_mask_node.add_child(object_mask_id_node)

        object_node.add_child(position_node, optional=True)
        object_node.add_child(adjective_node, optional=True)

        # build node dict
        adjust_node.build_node_dict()
        return adjust_node

    def create_undo(self):
        """
        Dependency tree:
            undo
        """
        undo_node = IntentNode(self.Intents.UNDO)
        undo_node.build_node_dict()
        return undo_node

    def create_redo(self):
        """
        Dependency tree:
            redo
        """
        redo_node = IntentNode(self.Intents.REDO)
        redo_node.build_node_dict()
        return redo_node

    def create_close(self):
        """
        Dependency tree:
            close
        """
        close_node = IntentNode(self.Intents.CLOSE)
        close_node.build_node_dict()
        return close_node
