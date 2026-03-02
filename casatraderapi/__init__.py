"""A python wrapper for CasaTrader API."""

import logging

def _prepare_logging():
    """Prepare logger for module CasaTrader API."""
    logger = logging.getLogger(__name__)
    #https://github.com/Lu-Yi-Hsun/casatraderapi_private/issues/1
    #try to fix this problem
    #logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.NullHandler())

    websocket_logger = logging.getLogger("websocket")
    websocket_logger.setLevel(logging.DEBUG)
    websocket_logger.addHandler(logging.NullHandler())

_prepare_logging()
