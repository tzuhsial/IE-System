"""
    Will expand this into a class in the future
"""

# User action_type and arguments
Ontology = {
    'open': ['action_type', 'image_path'],
    'load': ['action_type', 'b64_img_str'],
    'adjust': ['action_type', 'attribute', 'adjustValue', 'object'],
    'select_object': ['action_type', 'object'],
    'select_object_mask_id': ['action_type', 'object_mask_id'],
    'undo': ['action_type'],
    'redo': ['action_type'],
}
