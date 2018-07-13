"""
    Actions and corresponding list of arguments
"""
Ontology = {
    'open': ['action_type', 'image_path'],
    'load': ['action_type', 'b64_img_str'],
    'adjust': ['action_type', 'attribute', 'adjustValue', 'select'],
    'select': ['action_type', 'select'],
    'undo': ['action_type'],
    'redo': ['action_type'],
}
