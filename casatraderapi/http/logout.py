"""Module for CasaTrader http login resource."""

from casatraderapi.http.resource import Resource


class Logout(Resource):
    """Class for CasaTrader login resource."""
    # pylint: disable=too-few-public-methods

    url = ""

    def _post(self, data=None, headers=None):
        """Send get request for CasaTrader API login http resource.

        :returns: The instance of :class:`requests.Response`.
        """
        return self.api.send_http_request_v2(method="POST", url="https://auth.trade.casatrade.com/api/v1.0/logout",data=data, headers=headers)

    def __call__(self):
       
        return self._post()

