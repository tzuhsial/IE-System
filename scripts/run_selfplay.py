from configparser import ConfigParser
import logging
import json
import os
import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, root_dir)

from tqdm import tqdm

from iedsp import util
from iedsp.channel import ChannelPortal
from iedsp.photoshop import PhotoshopPortal
from iedsp.system import System
from iedsp.user import UserPortal
from iedsp.world import SelfPlayWorld

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def main(argv):
    # Get arguments
    config_file = argv[1]

    # Configsc
    config = ConfigParser(allow_no_value=True)
    config.read(config_file)

    agendas = util.load_from_pickle(config['DEFAULT']['AGENDA_PICKLE'])

    # Build Interaction World
    user = UserPortal(config['USER'])
    channel = ChannelPortal(config)
    system = System(config)  # system needs global_config
    photoshop = PhotoshopPortal(config['PHOTOSHOP'])

    agents = [user, channel, system, photoshop]

    world = SelfPlayWorld(agents)

    for agenda in tqdm(agendas):
        world.reset()
        user.load_agenda(agenda)

        while True:

            world.parley()

            if world.episode_done():
                break

        import pdb
        pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
