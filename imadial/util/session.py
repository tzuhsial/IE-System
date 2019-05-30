import logging
import os
import sys

import pickle

from .io import save_to_pickle, load_from_pickle

logger = logging.getLogger(__name__)


class SessionManager(object):
    """
    Defines basic methods
    """

    def __init__(self):
        raise NotImplementedError

    def create_session(self):
        raise NotImplementedError

    def retrieve(self, session_id):
        raise NotImplementedError

    def add_turn(self, session_id, system_state):
        raise NotImplementedError

    def add_survey(self, session_id, survey):
        raise NotImplementedError


class PickleManager(SessionManager):
    """
    Uses file storage to store serialized sessions
    """

    def __init__(self, pickle_dir):
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        self.pickle_dir = pickle_dir

    def get_session_path(self, session_id):
        session_path = os.path.join(
            self.pickle_dir, 'session.{}.pickle'.format(session_id))
        return session_path

    def create_session(self, session_id):
        doc = {
            "session_id": session_id,
            "image_id": -1,
            "manager": {},
            "imageeditengine": {},
            "acts": list()
        }
        session_path = self.get_session_path(session_id)
        save_to_pickle(doc, session_path)
        return doc

    def retrieve(self, session_id):
        session_path = self.get_session_path(session_id)
        if not os.path.exists(session_path):
            self.create_session(session_id)

        doc = load_from_pickle(session_path)

        last_system_state = {
            "manager": doc['manager'],
            "imageeditengine": doc['imageeditengine'],
            "acts": doc['acts'][-1]
        }
        return last_system_state

    def add_image_id(self, session_id, image_id):
        session_path = self.get_session_path(session_id)
        doc = load_from_pickle(session_path)
        doc['image_id'] = image_id
        save_to_pickle(doc, session_path)
        return doc

    def add_turn(self, session_id, system_state):
        session_path = self.get_session_path(session_id)
        doc = load_from_pickle(session_path)
        doc["manager"] = system_state['manager']
        doc['imageeditengine'] = system_state['imageeditengine']
        doc["acts"].append(system_state['acts'])
        save_to_pickle(doc, session_path)
        return doc

    def add_survey(self, session_id, survey):
        session_path = self.get_session_path(session_id)
        doc = load_from_pickle(session_path)
        doc["survey"] = survey
        save_to_pickle(doc, session_path)
        return doc
