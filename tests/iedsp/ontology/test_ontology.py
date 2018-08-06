
from iedsp.ontology import ImageEditOntology


def test_create():
    """
    Test whether we can successfully create intent tree, for now.
    """
    ontology = ImageEditOntology({})

    for intent in ["unknown", "open", "load", "close", "select_object", "adjust", "undo", "redo"]:
        tree = ontology.create(intent)
        assert tree.name == intent
