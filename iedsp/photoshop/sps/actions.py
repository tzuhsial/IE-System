

class PSAct:
    """
    Defines actions that are supported by photoshop
    """
    class Control:
        OPEN = "open"
        LOAD = "load"
        CLOSE = "close"
        REDO = "redo"
        UNDO = "undo"
        LOAD_MASK_STRS = "load_mask_strs"
        SELECT_OBJECT = "select_object"
        SELECT_OBJECT_MASK_ID = "select_object_mask_id"
        DESELECT = "deselect"

    class Edit:
        ADJUST = "adjust"
        ADJUST_COLOR = "adjust_color"


class PSArgs:
    """
    Arguments used by actions
    """
    IMAGE_PATH = "image_path"
    B64_IMG_STR = "b64_img_str"
    MASK_STRS = "mask_strs"
    OBJECT = "object"
    OBJECT_MASK_ID = "object_mask_id"

    ATTRIBUTE = "attribute"
    ADJUST_VALUE = "adjust_value"
    COLOR = "color"
