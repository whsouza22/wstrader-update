"""Module for CasaTrader appinit http resource."""

from casatraderapi.http.resource import Resource


class Appinit(Resource):
    """Class for CasaTrader login resource."""
    # pylint: disable=too-few-public-methods

    url = "appinit"

    def _get(self, data=None, headers=None):
        """Send get request for CasaTrader API appinit http resource.

        :returns: The instance of :class:`requests.Response`.
        """
        return self.send_http_request("GET", data=data, headers=headers)

    def __call__(self):
        """Method to get CasaTrader API appinit http request.

        :returns: The instance of :class:`requests.Response`.
        """
        return self._get()
