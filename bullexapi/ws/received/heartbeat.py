"""Module for Bullex websocket."""
import logging
import time

def heartbeat(api, message):
    if message["name"] == "heartbeat":
        try:
            msg = message["msg"]
            # msg pode ser int/float (timestamp) ou dict com 'heartbeatTime'
            if isinstance(msg, dict):
                hb_time = msg.get("heartbeatTime", int(time.time()))
            else:
                hb_time = msg
            api.heartbeat(hb_time)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Heartbeat response failed: {e}")