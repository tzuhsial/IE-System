from . import util
from .core import SystemAct
from .imageeditengine import ImageEditEnginePortal
from .manager import ManagerPortal
from .nlu import NLUPortal


class ImageEditRealUserInterface(object):
    """
    Interface to Real User
    does not contain Vision Engine

    Attributes
    - nlu
    - manager
    - imageeditengine

    """

    def __init__(self, config_file):
        """
        Set config and build agents
        Args:
            config_file (str)
        """
        # Load config
        config = util.load_from_json(config_file)
        agents_config = config["agents"]

        # Create modules
        self.nlu = NLUPortal(agents_config["system"]["nlu"])
        self.manager = ManagerPortal(agents_config["system"]["manager"])
        self.imageeditengine = ImageEditEnginePortal(
            agents_config["imageeditengine"])

    def reset(self):
        self.nlu.reset()
        self.manager.reset()
        self.imageeditengine.reset()

        self.acts = [{}] * 4  # first one for user

    def open(self, image_path):
        """ Open an image
        """
        self.reset()
        self.imageeditengine.open(image_path)
        self.acts[3] = self.imageeditengine.act()

    def get_image(self):
        """
        Gets image in numpy array form from System
        """
        return self.imageeditengine.get_image()

    def observe(self, observation):
        """
        observation comes from user or vision engine
        """
        self.observation = observation

    def act(self):
        """
        Args:
            usr_act (dict): user action
        Returns:
            sys_act (dict) : system action
        """
        # Order:
        # user, nlu, system, imageeditengine

        ################
        #   User Act   #
        ################
        self.acts[0] = self.observation

        #############################
        #    NLU or Vision Engine   #
        #############################
        if 'user_utterance' in self.acts[0]:
            usr_act = self.acts[0]
            self.nlu.observe(usr_act)
            nlu_or_vis_act = self.nlu.act()
        else:
            nlu_or_vis_act = self.acts[0]

        self.acts[1] = nlu_or_vis_act

        ###################
        #   Manager Act   #
        ###################
        imageeditengine_act = self.acts[3]
        self.manager.observe(imageeditengine_act)
        self.manager.observe(nlu_or_vis_act)
        system_act = self.manager.act()
        self.acts[2] = system_act

        ##########################
        #   imageeditengine Act  #
        ##########################
        self.imageeditengine.observe(system_act)
        imageeditengine_act = self.imageeditengine.act()
        self.acts[3] = imageeditengine_act

        ############################
        #   Reset Upon Execution   #
        ############################
        sys_dialogue_act = system_act['system_acts'][0]['dialogue_act']['value']
        if sys_dialogue_act == SystemAct.EXECUTE:
            self.manager.flush()  # Reset state

        return system_act

    def to_json(self):
        obj = {
            'acts': self.acts,
            'manager': self.manager.to_json(),
            'imageeditengine': self.imageeditengine.to_json()
        }
        return obj

    def from_json(self, obj):
        self.acts = obj['acts']
        self.manager.from_json(obj['manager'])
        self.imageeditengine.from_json(obj['imageeditengine'])
