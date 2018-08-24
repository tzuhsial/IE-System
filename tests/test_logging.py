import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.addHandler(logging.StreamHandler())

logger.info("info1")
logger.debug("debug1")
