import copy

import requests


class schema:
    """
        Provides schema of currently supported features by Photoshop backend
        Currently, we use SimplePhotoshop https://git.corp.adobe.com/tzlin/SimplePhotoshop
    """
    def default_goal_factory():
        return {}


class PhotoshopAPI(object):
    def __init__(self, api_url):
        """Stores the api for connection
        """
        self.api_url = api_url

    @property
    def name(self):
        return self.__class__.__name__

    def reset(self):
        self._status = schema.default_goal_factory()
        self.observe({})

    @property
    def status(self):
        """
            Json Representation
        """
        self._status

    def observe(self, observation):
        self.observation = observation

    def act(self):
        """
            Execute action on the image,
            action_type: 
                1. adjust
                2. non_adjust
        """

        # Get all system actions
        system_acts = self.observation.get('system_acts', list())
        for sys_act in system_acts:

            # Get parameters
            dialogue_act_type = sys_act.get('type')
            action_type = sys_act.get('action_type', 'null')
            slot_type = sys_act.get('type')
            slot_name = sys_act.get('slot')
            slot_value = sys_act.get('value')

            # Execute actions
            if action_type == 'null':
                pass
            elif dialogue_act_type == "inform":
                if slot_type == "adjust":
                    self._status[slot_name] += slot_value
                elif slot_type == "non_adjust":
                    self._status[slot_name] = slot_value
            else:
                raise NotImplementedError(
                    "Unknown dialogue_act_type \"{}\"".format(dialogue_act_type))

        photoshop_act = {}
        photoshop_act['image_state'] = copy.deepcopy(self._status)
        return photoshop_act
