"""
An ontology is used for two purposes
1. Generate simulated user goals
    - Provides easy access to schema
2. Construct dialogue state
    - Provides convenient functions calls
"""


class OntologyUnit(object):
    def __init__(self):
        raise NotImplementedError

    def to_state(self):
        raise NotImplementedError


class ImageEditSlot(OntologyUnit):
    def __init__(self, slot_type, name, threshold, possible_values=[], permit_new_value=True):
        self.slot_type = slot_type
        self.name = name
        self.threshold = threshold
        self.possible_values = possible_values
        self.permit_new = permit_new_value

    def getPossibleValues(self):
        return self.possible_values


class ImageEditDomain(OntologyUnit):
    def __init__(self, edit_type, name, slots=[]):
        self.edit_type = edit_type
        self.name = name
        self.slots = slots

    def getSlotNames(self):
        return [slot.name for slot in self.slots]


class ImageEditOntology(OntologyUnit):
    """
    Ontology for Image Editing
    """
    class Domains:
        OPEN = "open"
        LOAD = "load"
        CLOSE = "close"
        ADJUST = "adjust"
        UNDO = "undo"
        REDO = "redo"
        OOD = "out_of_domain"

    class DomainTypes:
        CONTROL = 'control'
        EDIT = 'edit'

    class Slots:
        IMAGE_PATH = 'image_path'
        B64_IMG_STR = 'b64_img_str'
        ATTRIBUTE = 'attribute'
        ADJUST_VALUE = 'adjust_value'
        OBJECT = 'object'
        OBJECT_MASK = 'object_mask'
        OBJECT_MASK_ID = 'object_mask_id'

    class SlotTypes:
        BELIEF_SLOT = 'BeliefSlot'
        PSTOOL_SLOT = 'PSToolSlot'

    def __init__(self, config={}):
        """
        ***Important****
        Here we define the dependencies of the domain and slots
        """
        # Get default slot threshold

        slot_thresh = config.get('SLOT_DEFAULT_THRESHOLD', 0.7)
        #########################
        #      Build Slots      #
        #########################

        # Slot: image_path
        image_path_thresh = config.get("IMAGE_PATH_THRESHOLD", slot_thresh)
        image_path_slot = ImageEditSlot(
            self.SlotTypes.PSTOOL_SLOT, self.Slots.IMAGE_PATH, image_path_thresh)

        # Slot: b64_img_str
        b64_img_str_thresh = config.get("B64_IMG_STR_THRESHOLD", slot_thresh)
        b64_img_str_slot = ImageEditSlot(
            self.SlotTypes.PSTOOL_SLOT, self.Slots.B64_IMG_STR, b64_img_str_thresh)

        # Slot: attribute
        attribute_thresh = config.get("ATTRIBUTE_THRESHOLD", slot_thresh)
        attribute_values = ["brightness", "contrast",
                            "hue", "saturation", "lightness"]
        attribute_slot = ImageEditSlot(
            self.SlotTypes.BELIEF_SLOT, self.Slots.ATTRIBUTE, attribute_thresh, attribute_values, False)

        # Slot: adjust_value
        adjust_value_thresh = config.get("ADJUST_VALUE_THRESHOLD", slot_thresh)
        adjust_value_values = [-40, -25, -10, 10, 25, 40]
        adjust_value_slot = ImageEditSlot(
            self.SlotTypes.BELIEF_SLOT, self.Slots.ADJUST_VALUE, adjust_value_thresh, adjust_value_values)

        # Slot: object
        object_thresh = config.get("OBJECT_THRESHOLD", slot_thresh)
        object_slot = ImageEditSlot(
            self.SlotTypes.BELIEF_SLOT, self.Slots.OBJECT, object_thresh)

        # Build slot dictionary
        self.slots = {
            image_path_slot.name: image_path_slot,
            b64_img_str_slot.name: b64_img_str_slot,
            attribute_slot.name: attribute_slot,
            adjust_value_slot.name: adjust_value_slot,
            object_slot.name: object_slot,
        }

        ##########################
        #      Build Domains     #
        ##########################

        # Domain: open
        open_slots = [image_path_slot]
        open_domain = ImageEditDomain(
            self.DomainTypes.CONTROL, self.Domains.OPEN, open_slots)

        # Domain: load
        load_slots = [b64_img_str_slot]
        load_domain = ImageEditDomain(
            self.DomainTypes.CONTROL, self.Domains.LOAD, load_slots)

        # Domain: adjust
        adjust_slots = [attribute_slot, adjust_value_slot, object_slot]
        adjust_domain = ImageEditDomain(
            self.DomainTypes.EDIT, self.Domains.ADJUST, adjust_slots)

        # Domain: redo
        redo_slots = []
        redo_domain = ImageEditDomain(
            self.DomainTypes.CONTROL, self.Domains.REDO)

        # Domain: undo
        undo_slots = []
        undo_domain = ImageEditDomain(
            self.DomainTypes.CONTROL, self.Domains.UNDO)

        # Build slot dictionary
        self.domains = {
            open_domain.name: open_domain,
            load_domain.name: load_domain,
            adjust_domain.name: adjust_domain,
            redo_domain.name: redo_domain,
            undo_domain.name: undo_domain
        }

    def getDomainWithName(self, domain_name):
        return self.domains.get(domain_name, None)

    def getSlotWithName(self, slot_name):
        return self.slots.get(slot_name, None)


def getOntology(ontology_config):
    """
    Returns Ontology Class with config
    Args:
        ontology (str): name of ontology
    Returns:
        class: corresponding ontology class
    """
    if config["ONTOLOGY"] == "ImageEditOntology":
        return ImageEditOntology(config)
    else:
        raise ValueError("Unknown ontology: {}".format(ontology))


if __name__ == "__main__":
    import configparser
    config = configparser.ConfigParser()
    config.read('../config.dev.ini')

    ontology = ImageEditOntology(config['ONTOLOGY'])
    import pdb
    pdb.set_trace()
