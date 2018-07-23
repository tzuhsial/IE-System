
from iedsp.ontology import ImageEditOntology
from iedsp.system.dialoguestate import BeliefSlot, PSToolSlot, DialogueState


def slot_dict_builder(slot, value, conf):
    return {"slot": slot, "value": value, "conf": conf}


def test_Slots():

    adjustValue = ["more", "less"]

    slot = BeliefSlot("adjustValue", adjustValue, False)

    assert slot.addNewObservation("more", 0.8, 1)
    assert not slot.addNewObservation("a little more", 0.8, 2)
    assert slot.getMaxConfValue() == "more"

    slot = PSToolSlot('image_path')
    assert slot.addNewObservation("/example/path/image.jpg", 1) == True


def test_DialogueState():

    dialogueState = DialogueState(ImageEditOntology)

    domain = "adjust"
    slots = [slot_dict_builder(
        "attribute", "brightness", 0.8), slot_dict_builder("object", "dog", 0.7)]
    turn_id = 1

    assert dialogueState.update(domain, slots, turn_id) == True



    import pdb
    pdb.set_trace()
