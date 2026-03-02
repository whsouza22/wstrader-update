"""Module for CasaTrader websocket."""

def financial_information(api, message):
    if message["name"] == "financial-information":
            api.financial_information = message