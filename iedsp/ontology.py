"""
    Defines the Ontology
"""


class Ontology:
    """
        key: intent
        value: argument list
    """
    arguments = {
        'open': ['image_path'],
        'load': ['b64_img_str'],
        'adjust': ['attribute', 'adjustValue', 'object'],
        'select_object': ['object'],
        'select_object_mask_id': ['object_mask_id'],
        'undo': [],
        'redo': [],
        'close': [],
    }

    # Supported Attributes
    attributes = [
        'brightness', 'contrast', 'hue', 'saturation', 'lightness'
    ]

    def getArgumentsWithIntent(intent):
        """Get argument list with intent as key
        """
        if intent in Ontology.arguments:
            return Ontology.arguments.get(intent)
        else:
            return []

    def getPossibleValuesWithSlot(slot):
        """
        """
        if slot == 'attribute':
            return Ontology.attributes
        elif slot == "adjustValue":
            return list(range(-100, 101))
        else:
            raise ValueError("Unknown slot: {}".format(slot))

if __name__ == "__main__":
    assert Ontology.getArgumentsWithIntent('open') == ['image_path']
