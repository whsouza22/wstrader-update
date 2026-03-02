"""Module for CasaTrader http login resource."""

from casatraderapi.http.resource import Resource


class Login2FA(Resource):
    """Class for CasaTrader login resource."""
    # pylint: disable=too-few-public-methods

    url = ""

    def _post(self, data=None, headers=None):
        """Send get request for CasaTrader API login http resource.

        :returns: The instance of :class:`requests.Response`.
        """
        return self.api.send_http_request_v2(method="POST", url="https://auth.trade.casatrade.com/api/v2/login",data=data, headers=headers)

    def __call__(self, username, password, token_login):
        """Method to get CasaTrader API login http request.

        :param str username: The username of a CasaTrader server.
        :param str password: The password of a CasaTrader server.
        :param str token: The token of a CasaTrader server 2FA.

        :returns: The instance of :class:`requests.Response`.
        """
        data = {"identifier": username,
                "password": password,
                "token": token_login}

        return self._post(data=data)
