"""
    An interface to build photoshop related apis.
    This file also includes the schema used
"""
import copy

# Demo 1
adjust_slots = ["slot_1"]
non_adjust_slots = []
non_adjust_slots_default = {}


class PhotoshopAPI(object):
    def __init__(self):
        """
            Connection with Photoshop
        """
        pass

    @property
    def name(self):
        return self.__class__.__name__

    def reset(self):
        # Reset status
        status = {}
        for slot in adjust_slots:
            status[slot] = 0.
        for slot in non_adjust_slots:
            status[slot] = non_adjust_slots_default.get(slot)
        self._status = status

        # Initial observation
        observation = {}
        self.observe(observation)

    @property
    def status(self):
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
            slot = sys_act.get('slot', 'null')
            value = sys_act.get('value', 'null')

            # Execute actions
            if dialogue_act_type == "request" or action_type == 'null':
                pass
            elif dialogue_act_type == "inform":
                if action_type == "adjust":
                    self._status[slot] += value
                elif action_type == "non_adjust":
                    self._status[slot] = value
            else:
                raise NotImplementedError(
                    "Unknown dialogue_act_type \"{}\"".format(dialogue_act_type))

        photoshop_act = {}
        photoshop_act['image_state'] = copy.deepcopy(self._status)
        return photoshop_act
