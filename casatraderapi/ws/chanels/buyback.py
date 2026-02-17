"""Module for CasaTrader buyback websocket chanel."""

from casatraderapi.ws.chanels.base import Base


class Buyback(Base):
    """Class for CasaTrader subscribe to buyback websocket chanel."""
    # pylint: disable=too-few-public-methods

    name = "buyback"

    def __call__(self):
        """Method to send message to buyback websocket chanel."""
        pass
