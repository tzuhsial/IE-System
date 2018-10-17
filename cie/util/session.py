import logging
import sys

from pymongo import MongoClient
from tinydb import TinyDB, Query

logger = logging.getLogger(__name__)


def SessionPortal(session_config):
    session_type = session_config["session"]
    return builder(session_type)(session_config)


class SessionManager(object):
    def __init__(self, host=None, port=None, **kwargs):
        raise NotImplementedError

    def retrieve(self, session_id):
        raise NotImplementedError

    def add_turn(self, session_id, system_state, photoshop_state, turn_info):
        raise NotImplementedError

    def add_result(self, session_id, result):
        raise NotImplementedError

    def _init_dialogue(self, session_id):
        doc = {
            "session_id": session_id,
            "system_state": {},
            "photoshop_state": {},
            "turns": list()
        }
        return doc


class MongoDBManager(SessionManager):
    def __init__(self, host=None, port=None, **kwargs):
        # self.client = MongoClient(host, port)
        self.client = MongoClient()
        self.db = self.client["cie"]
        self.dialogues = self.db["dialogues"]

    def retrieve(self, session_id):
        session_key = {"session_id": session_id}
        if self.dialogues.count_documents(session_key) == 0:
            doc = self._init_dialogue(session_id)
            self.dialogues.insert_one(doc)
        return self.dialogues.find_one(session_key)

    def add_turn(self, session_id, system_state, photoshop_state, turn_info):

        doc = self.retrieve(session_id)
        doc["system_state"] = system_state
        doc["photoshop_state"] = photoshop_state
        doc["turns"].append(turn_info)

        self.dialogues.replace_one({"session_id": session_id}, doc)

    def add_policy(self, session_id, policy):
        key = {"session_id": session_id}
        self.dialogues.update_one(
            key,
            {
                "$set": {
                    "policy": policy
                }
            }
        )

    def add_result(self, session_id, result):
        key = {"session_id": session_id}
        self.dialogues.update_one(
            key,
            {
                "$set": {
                    "result": result
                }
            }
        )


class TinyDBManager(SessionManager):
    def __init__(self, db_path, **kwargs):
        self.db = TinyDB(db_path)

    def retrieve(self, session_id):
        session = Query()
        dialogues = self.db.search(session.id == session_id)
        if len(dialogues) == 0:
            doc = self._init_dialogue()
            self.db.insert(doc)
        else:
            doc = dialogues[0]
        return doc

    def add_turn(self, session_id, state, photoshop, turn_info):
        dialogue = self.retrieve(session_id)
        dialogue["state"] = state
        dialogue["photoshop"] = photoshop
        dialogue["turns"].append(turn_info)
        session = Query()
        self.db.upsert(dialogue, session.id == session_id)


def builder(string):
    """
    Gets node class with string
    """
    try:
        return getattr(sys.modules[__name__], string)
    except AttributeError:
        logger.error("Unknown node: {}".format(string))
        return None


if __name__ == "__main__":
    pass
