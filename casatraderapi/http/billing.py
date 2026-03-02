"""Module for CasaTrader billing resource."""

from casatraderapi.http.resource import Resource


class Billing(Resource):
    """Class for CasaTrader billing resource."""
    # pylint: disable=too-few-public-methods

    url = "billing"
