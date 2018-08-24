import configparser


def test_config_format():
    """ Test config format
    """
    config = configparser.ConfigParser()
    config.read('config.dev.ini')
