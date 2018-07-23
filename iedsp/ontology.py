class ImageEditOntology:
    """
        Ontology for Image Editing
        Every domain is an Image Edit "intent"
        Every domain have corresponding slots
        Attributes:
            domain_table (dict): domain to slot list mapping
            slot_class_map (dict): slot to class string mapping
            domain: domain as list
            slots: slot as list
    """
    OPEN = "open"
    LOAD = "load"
    ADJUST = "adjust"
    SELECT_OBJECT = "select_object"
    SELECT_OBJECT_MASK_ID = "select_object_mask_Id"
    UNDO = "undo"
    REDO = "redo"
    CLOSE = "close"

    OOD = "out_of_domain"

    domain_slot_map = {
        'open': ['image_path'],
        'load': ['b64_img_str'],
        'adjust': ['attribute', 'adjustValue', 'object'],
        'select_object': ['object'],
        'select_object_mask_id': ['object_mask_id'],
        'undo': [],
        'redo': [],
        'close': [],
    }

    domain_type_map = {
        'open': 'control',
        'load': 'control',
        'adjust': 'edit',
        'select_object': 'control',
        'select_object_mask_id': 'control',
        'undo': 'control',
        'redo': 'control',
        'close': 'control',
    }

    slot_class_map = {
        "image_path": "PSToolSlot",
        "b64_img_str": "PSToolSlot",
        "attribute": "BeliefSlot",
        "adjustValue": "BeliefSlot",
        "object": "BeliefSlot",
        "object_mask_id": "BeliefSlot",
    }


def getOntologyByName(ontology):
    """
    Returns Ontology Class with Name
    Args:
        ontology (str): name of ontology
    Returns:
        class: corresponding ontology class
    """
    if ontology == "ImageEditOntology":
        return ImageEditOntology
    else:
        raise ValueError("Unknown ontology: {}".format(ontology))


if __name__ == "__main__":
    pass
