def find_slot_with_key(key, slots):
    """
    """
    for idx, slot_dict in enumerate(slots):
        if slot_dict['slot'] == key:
            return idx, slot_dict
    return -1, None