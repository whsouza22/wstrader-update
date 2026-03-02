"""Module for CasaTrader websocket."""

def order_canceled(api, message):
    if message["name"] == "order-canceled":
        api.order_canceled = message