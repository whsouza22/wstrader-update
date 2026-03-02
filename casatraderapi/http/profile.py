"""Module for CasaTrader profile resource."""

from casatraderapi.http.resource import Resource


class Profile(Resource):
    """Class for CasaTrader profile resource."""
    # pylint: disable=too-few-public-methods

    url = "profile"
