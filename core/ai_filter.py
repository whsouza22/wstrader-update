def ai_confirms(side: str, p_up: float, min_conf: float):
    conf = max(p_up, 1.0 - p_up)
    if conf < min_conf:
        return False, "LOW_AI_CONF", conf

    if side == "BUY" and p_up < 0.55:
        return False, "AI_NOT_ALIGNED", conf
    if side == "SELL" and p_up > 0.45:
        return False, "AI_NOT_ALIGNED", conf

    return True, "AI_OK", conf
