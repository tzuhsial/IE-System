"""
    Will expand this into a class in the future
"""

# User action_type and arguments
# Intents & Slots
Ontology = {
    'open': ['intent', 'image_path'],
    'load': ['intent', 'b64_img_str'],
    'adjust': ['intent', 'attribute', 'adjustValue', 'object'],
    'select_object': ['intent', 'object'],
    'select_object_mask_id': ['intent', 'object_mask_id'],
    'undo': ['intent'],
    'redo': ['intent'],
}
