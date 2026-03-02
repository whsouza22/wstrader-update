"""Module for CasaTrader http auth resource."""

from casatraderapi.http.resource import Resource


class Auth(Resource):
    """Class for CasaTrader http auth resource."""
    # pylint: disable=too-few-public-methods

    url = "auth"
