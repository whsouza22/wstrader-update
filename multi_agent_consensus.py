"""
Sistema de Consenso Multi-Agente

4 Agentes de IA que analisam o mercado de perspectivas diferentes.
SO ENTRA QUANDO OS 4 CONCORDAM NA MESMA DIRECAO.

Agentes:
1. Agente S/R - Analisa Suporte/Resistencia e toques
2. Agente Tendencia - Analisa tendencia local (ultimas velas)
3. Agente Padroes - Analisa padroes de velas (reversao, continuacao)
4. Agente CNN - Rede neural que prevê direcao

Votacao:
- CALL: Todos os 4 agentes votam CALL
- PUT: Todos os 4 agentes votam PUT
- NO_TRADE: Qualquer divergencia = nao opera
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Voto(Enum):
    CALL = "CALL"
    PUT = "PUT"
    NEUTRO = "NEUTRO"


@dataclass
class ResultadoAgente:
    """Resultado da analise de um agente."""
    nome: str
    voto: Voto
    confianca: float  # 0.0 a 1.0
    motivo: str


class AgenteBase:
    """Classe base para todos os agentes."""

    def __init__(self, nome: str):
        self.nome = nome

    def analisar(self, df_m1: pd.DataFrame, atr_val: float, **kwargs) -> ResultadoAgente:
        raise NotImplementedError


# ============================================================================
#                         AGENTE 1: SUPORTE/RESISTENCIA
# ============================================================================

class AgenteSR(AgenteBase):
    """
    Agente especialista em Suporte e Resistencia COM CONTEXTO DE TENDENCIA.

    REGRA IMPORTANTE:
    - Em DOWNTREND: So opera PUT (resistencia ou rompimento de suporte)
    - Em UPTREND: So opera CALL (suporte ou rompimento de resistencia)
    - Em LATERAL: Opera reversao normal (suporte=CALL, resistencia=PUT)

    Logica:
    - DOWNTREND + toque resistencia = PUT (rejeicao, continua caindo)
    - DOWNTREND + toque suporte = NEUTRO ou PUT (suporte vai romper)
    - UPTREND + toque suporte = CALL (rejeicao, continua subindo)
    - UPTREND + toque resistencia = NEUTRO ou CALL (resistencia vai romper)
    """

    def __init__(self):
        super().__init__("S/R")

    def _detect_trend(self, df: pd.DataFrame) -> str:
        """Detecta tendencia baseado em topos e fundos."""
        if len(df) < 20:
            return "LATERAL"

        # Pega os highs e lows das ultimas 20 velas
        ultimas = df.tail(20)
        highs = [r["high"] for _, r in ultimas.iterrows()]
        lows = [r["low"] for _, r in ultimas.iterrows()]

        # Compara primeira metade com segunda metade
        first_half_high = max(highs[:10])
        second_half_high = max(highs[10:])
        first_half_low = min(lows[:10])
        second_half_low = min(lows[10:])

        # Conta velas verdes
        verdes = sum(1 for _, r in ultimas.iterrows() if r["close"] > r["open"])

        # DOWNTREND: Topos mais baixos E fundos mais baixos
        if second_half_high < first_half_high and second_half_low < first_half_low:
            return "DOWN"

        # UPTREND: Topos mais altos E fundos mais altos
        if second_half_high > first_half_high and second_half_low > first_half_low:
            return "UP"

        # Usa contagem de velas como fallback
        if verdes >= 13:
            return "UP"
        elif verdes <= 7:
            return "DOWN"

        return "LATERAL"

    def analisar(self, df_m1: pd.DataFrame, atr_val: float, sr_levels: Dict = None, **kwargs) -> ResultadoAgente:
        if len(df_m1) < 50:
            return ResultadoAgente(self.nome, Voto.NEUTRO, 0.0, "poucas_velas")

        atr_safe = max(atr_val, 1e-9)
        last = df_m1.iloc[-1]
        candle_high = float(last["high"])
        candle_low = float(last["low"])
        candle_close = float(last["close"])
        candle_open = float(last["open"])

        # DETECTA TENDENCIA PRIMEIRO
        trend = self._detect_trend(df_m1)

        # Se nao tiver niveis S/R, vota na direcao da tendencia
        if not sr_levels:
            if trend == "DOWN":
                return ResultadoAgente(self.nome, Voto.PUT, 0.45, f"sem_sr_trend_{trend}")
            elif trend == "UP":
                return ResultadoAgente(self.nome, Voto.CALL, 0.45, f"sem_sr_trend_{trend}")
            return ResultadoAgente(self.nome, Voto.NEUTRO, 0.0, "sem_niveis_sr")

        resistencias = sr_levels.get("resistencias", [])
        suportes = sr_levels.get("suportes", [])

        tolerancia = atr_safe * 0.25
        melhor_suporte = None
        melhor_resistencia = None

        # Procura suporte tocado
        for sup_lvl, toques in suportes:
            if toques >= 2:
                dist = abs(candle_low - sup_lvl)
                if dist <= tolerancia and candle_close > sup_lvl:
                    if melhor_suporte is None or toques > melhor_suporte[1]:
                        melhor_suporte = (sup_lvl, toques, dist)

        # Procura resistencia tocada
        for res_lvl, toques in resistencias:
            if toques >= 2:
                dist = abs(candle_high - res_lvl)
                if dist <= tolerancia and candle_close < res_lvl:
                    if melhor_resistencia is None or toques > melhor_resistencia[1]:
                        melhor_resistencia = (res_lvl, toques, dist)

        # === DECISAO BASEADA EM TENDENCIA + S/R ===

        # DOWNTREND: So opera PUT
        if trend == "DOWN":
            if melhor_resistencia:
                # Resistencia em downtrend = otimo para PUT (rejeicao)
                conf = min(1.0, 0.6 + (melhor_resistencia[1] * 0.1))
                return ResultadoAgente(self.nome, Voto.PUT, conf,
                                      f"DOWN_resistencia_toques={melhor_resistencia[1]}")
            if melhor_suporte:
                # Suporte em downtrend = vai romper, PUT tambem
                conf = min(1.0, 0.5 + (melhor_suporte[1] * 0.05))
                return ResultadoAgente(self.nome, Voto.PUT, conf,
                                      f"DOWN_suporte_romper_toques={melhor_suporte[1]}")
            # Sem S/R mas em downtrend = PUT
            return ResultadoAgente(self.nome, Voto.PUT, 0.50, "DOWN_sem_sr")

        # UPTREND: So opera CALL
        if trend == "UP":
            if melhor_suporte:
                # Suporte em uptrend = otimo para CALL (rejeicao)
                conf = min(1.0, 0.6 + (melhor_suporte[1] * 0.1))
                return ResultadoAgente(self.nome, Voto.CALL, conf,
                                      f"UP_suporte_toques={melhor_suporte[1]}")
            if melhor_resistencia:
                # Resistencia em uptrend = vai romper, CALL tambem
                conf = min(1.0, 0.5 + (melhor_resistencia[1] * 0.05))
                return ResultadoAgente(self.nome, Voto.CALL, conf,
                                      f"UP_resistencia_romper_toques={melhor_resistencia[1]}")
            # Sem S/R mas em uptrend = CALL
            return ResultadoAgente(self.nome, Voto.CALL, 0.50, "UP_sem_sr")

        # LATERAL: Reversao normal
        if melhor_suporte and not melhor_resistencia:
            conf = min(1.0, 0.5 + (melhor_suporte[1] * 0.1))
            return ResultadoAgente(self.nome, Voto.CALL, conf,
                                  f"LATERAL_suporte_toques={melhor_suporte[1]}")

        if melhor_resistencia and not melhor_suporte:
            conf = min(1.0, 0.5 + (melhor_resistencia[1] * 0.1))
            return ResultadoAgente(self.nome, Voto.PUT, conf,
                                  f"LATERAL_resistencia_toques={melhor_resistencia[1]}")

        # Sem toque claro - vota na direcao da ultima vela
        if candle_close > candle_open:
            return ResultadoAgente(self.nome, Voto.CALL, 0.35, "sem_sr_vela_verde")
        elif candle_close < candle_open:
            return ResultadoAgente(self.nome, Voto.PUT, 0.35, "sem_sr_vela_vermelha")

        return ResultadoAgente(self.nome, Voto.NEUTRO, 0.0, "sem_toque_sr")


# ============================================================================
#                         AGENTE 2: TENDENCIA
# ============================================================================

class AgenteTendencia(AgenteBase):
    """
    Agente especialista em Tendencia.

    Analisa:
    - Direcao das ultimas N velas
    - Variacao percentual recente
    - Sequencia de cores (verde/vermelho)
    """

    def __init__(self, janela: int = 7):
        super().__init__("Tendencia")
        self.janela = janela

    def analisar(self, df_m1: pd.DataFrame, atr_val: float, **kwargs) -> ResultadoAgente:
        if len(df_m1) < self.janela + 1:
            return ResultadoAgente(self.nome, Voto.NEUTRO, 0.0, "poucas_velas")

        ultimas = df_m1.tail(self.janela)
        preco_inicio = float(df_m1.iloc[-self.janela]["close"])
        preco_fim = float(df_m1.iloc[-1]["close"])

        # Variacao percentual
        var_pct = ((preco_fim - preco_inicio) / preco_inicio) * 100 if preco_inicio > 0 else 0

        # Contagem de velas
        velas_alta = sum(1 for _, r in ultimas.iterrows() if r["close"] > r["open"])
        velas_baixa = self.janela - velas_alta

        # Tendencia de ALTA: 4+ velas verdes OU subiu (mais permissivo)
        if velas_alta >= 4 or var_pct > 0.03:
            conf = min(1.0, 0.5 + (velas_alta * 0.07) + (var_pct * 2))
            return ResultadoAgente(self.nome, Voto.CALL, conf, f"alta_velas={velas_alta}_var={var_pct:.2f}%")

        # Tendencia de BAIXA: 4+ velas vermelhas OU caiu (mais permissivo)
        if velas_baixa >= 4 or var_pct < -0.03:
            conf = min(1.0, 0.5 + (velas_baixa * 0.07) + (abs(var_pct) * 2))
            return ResultadoAgente(self.nome, Voto.PUT, conf, f"baixa_velas={velas_baixa}_var={var_pct:.2f}%")

        # Lateralizado - ainda tenta dar uma direcao baseado na ultima vela
        last = df_m1.iloc[-1]
        if last["close"] > last["open"]:
            return ResultadoAgente(self.nome, Voto.CALL, 0.4, f"leve_alta_var={var_pct:.2f}%")
        elif last["close"] < last["open"]:
            return ResultadoAgente(self.nome, Voto.PUT, 0.4, f"leve_baixa_var={var_pct:.2f}%")

        return ResultadoAgente(self.nome, Voto.NEUTRO, 0.3, f"lateral_var={var_pct:.2f}%")


# ============================================================================
#                         AGENTE 3: PADROES DE VELAS
# ============================================================================

class AgentePadroes(AgenteBase):
    """
    Agente especialista em Padroes de Velas.

    Analisa:
    - Pinbar (martelo, estrela cadente)
    - Engolfo
    - Doji
    - Pavios significativos
    """

    def __init__(self):
        super().__init__("Padroes")

    def analisar(self, df_m1: pd.DataFrame, atr_val: float, **kwargs) -> ResultadoAgente:
        if len(df_m1) < 3:
            return ResultadoAgente(self.nome, Voto.NEUTRO, 0.0, "poucas_velas")

        last = df_m1.iloc[-1]
        prev = df_m1.iloc[-2]

        o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
        po, ph, pl, pc = float(prev["open"]), float(prev["high"]), float(prev["low"]), float(prev["close"])

        corpo = abs(c - o)
        range_total = h - l if h > l else 0.0001
        pavio_sup = h - max(o, c)
        pavio_inf = min(o, c) - l

        corpo_frac = corpo / range_total
        pavio_sup_frac = pavio_sup / range_total
        pavio_inf_frac = pavio_inf / range_total

        # PINBAR DE ALTA (martelo): pavio inferior grande, corpo pequeno em cima
        if pavio_inf_frac > 0.5 and corpo_frac < 0.35 and c > o:
            return ResultadoAgente(self.nome, Voto.CALL, 0.75, "martelo_alta")

        # PINBAR DE BAIXA (estrela cadente): pavio superior grande, corpo pequeno embaixo
        if pavio_sup_frac > 0.5 and corpo_frac < 0.35 and c < o:
            return ResultadoAgente(self.nome, Voto.PUT, 0.75, "estrela_baixa")

        # ENGOLFO DE ALTA: vela atual verde engole a anterior vermelha
        if c > o and pc < po:  # atual verde, anterior vermelha
            if c > po and o < pc:  # engolfa
                return ResultadoAgente(self.nome, Voto.CALL, 0.70, "engolfo_alta")

        # ENGOLFO DE BAIXA: vela atual vermelha engole a anterior verde
        if c < o and pc > po:  # atual vermelha, anterior verde
            if c < po and o > pc:  # engolfa
                return ResultadoAgente(self.nome, Voto.PUT, 0.70, "engolfo_baixa")

        # Pavio de rejeicao simples (mais permissivo)
        if pavio_inf_frac > 0.30 and c > o:
            return ResultadoAgente(self.nome, Voto.CALL, 0.55, "rejeicao_baixo")

        if pavio_sup_frac > 0.30 and c < o:
            return ResultadoAgente(self.nome, Voto.PUT, 0.55, "rejeicao_cima")

        # Se nao tem padrao claro, vota baseado na cor da vela
        if c > o:
            return ResultadoAgente(self.nome, Voto.CALL, 0.45, "vela_verde")
        elif c < o:
            return ResultadoAgente(self.nome, Voto.PUT, 0.45, "vela_vermelha")

        return ResultadoAgente(self.nome, Voto.NEUTRO, 0.3, "doji")


# ============================================================================
#                         AGENTE 4: CNN (REDE NEURAL)
# ============================================================================

class AgenteCNN(AgenteBase):
    """
    Agente que usa a rede neural CNN para prever direcao.

    Usa o modelo TradingCNN existente para fazer previsoes.
    """

    def __init__(self, trading_cnn=None):
        super().__init__("CNN")
        self.trading_cnn = trading_cnn

    def set_model(self, trading_cnn):
        """Define o modelo CNN."""
        self.trading_cnn = trading_cnn

    def analisar(self, df_m1: pd.DataFrame, atr_val: float, **kwargs) -> ResultadoAgente:
        if self.trading_cnn is None:
            # Se CNN nao disponivel, vota baseado na ultima vela
            if len(df_m1) > 0:
                last = df_m1.iloc[-1]
                if last["close"] > last["open"]:
                    return ResultadoAgente(self.nome, Voto.CALL, 0.4, "sem_cnn_vela_verde")
                elif last["close"] < last["open"]:
                    return ResultadoAgente(self.nome, Voto.PUT, 0.4, "sem_cnn_vela_vermelha")
            return ResultadoAgente(self.nome, Voto.NEUTRO, 0.0, "cnn_nao_disponivel")

        try:
            # Usa o metodo predict do TradingCNN
            resultado = self.trading_cnn.predict(df_m1)

            classe = resultado.get("class", "NO_TRADE")
            prob = resultado.get("probability", 0.5)
            conf = resultado.get("confidence", 0.0)

            # Mais permissivo: aceita prob >= 0.4 (em vez de > 0.5)
            if classe == "CALL" and prob >= 0.4:
                return ResultadoAgente(self.nome, Voto.CALL, max(0.5, conf), f"cnn_prob={prob:.2f}")
            elif classe == "PUT" and prob >= 0.4:
                return ResultadoAgente(self.nome, Voto.PUT, max(0.5, conf), f"cnn_prob={prob:.2f}")
            else:
                # Se incerto, vota baseado na ultima vela
                if len(df_m1) > 0:
                    last = df_m1.iloc[-1]
                    if last["close"] > last["open"]:
                        return ResultadoAgente(self.nome, Voto.CALL, 0.4, f"cnn_incerto_vela_verde")
                    elif last["close"] < last["open"]:
                        return ResultadoAgente(self.nome, Voto.PUT, 0.4, f"cnn_incerto_vela_vermelha")
                return ResultadoAgente(self.nome, Voto.NEUTRO, 0.3, f"cnn_incerto_prob={prob:.2f}")

        except Exception as e:
            return ResultadoAgente(self.nome, Voto.NEUTRO, 0.0, f"cnn_erro={str(e)[:20]}")


# ============================================================================
#                         AGENTE 5: ESTRUTURA DE TENDENCIA
# ============================================================================

class AgenteEstrutura(AgenteBase):
    """
    Agente que analisa estrutura de tendencia (LH/HL + pullbacks).

    Detecta:
    - DOWNTREND: Topos mais baixos (LH) + Fundos mais baixos (LL)
    - UPTREND: Fundos mais altos (HL) + Topos mais altos (HH)
    - Pullbacks como oportunidades de entrada NA DIRECAO da tendencia

    Exemplo: Se topo atual < topo anterior e preco fez pullback = PUT
    """

    def __init__(self, swing_lookback: int = 5):
        super().__init__("Estrutura")
        self.swing_lookback = swing_lookback

    def _find_swings(self, df: pd.DataFrame) -> Tuple[list, list]:
        """Encontra topos e fundos."""
        if len(df) < self.swing_lookback * 2 + 1:
            return [], []

        highs = []  # (index, price)
        lows = []   # (index, price)
        lb = self.swing_lookback

        for i in range(lb, len(df) - lb):
            h = df.iloc[i]["high"]
            l = df.iloc[i]["low"]

            # Verifica topo
            is_high = all(df.iloc[j]["high"] < h for j in range(i-lb, i+lb+1) if j != i)
            if is_high:
                highs.append((i, h))

            # Verifica fundo
            is_low = all(df.iloc[j]["low"] > l for j in range(i-lb, i+lb+1) if j != i)
            if is_low:
                lows.append((i, l))

        return highs[-4:], lows[-4:]  # Ultimos 4 de cada

    def analisar(self, df_m1: pd.DataFrame, atr_val: float, **kwargs) -> ResultadoAgente:
        if len(df_m1) < 30:
            # Dados insuficientes - vota baseado na ultima vela
            if len(df_m1) > 0:
                last = df_m1.iloc[-1]
                if last["close"] > last["open"]:
                    return ResultadoAgente(self.nome, Voto.CALL, 0.4, "dados_insuf_vela_verde")
                elif last["close"] < last["open"]:
                    return ResultadoAgente(self.nome, Voto.PUT, 0.4, "dados_insuf_vela_vermelha")
            return ResultadoAgente(self.nome, Voto.NEUTRO, 0.3, "dados_insuficientes")

        highs, lows = self._find_swings(df_m1)

        if len(highs) < 2 or len(lows) < 2:
            # Poucos swings - vota baseado na tendencia simples
            ultimas = df_m1.tail(10)
            verdes = sum(1 for _, r in ultimas.iterrows() if r["close"] > r["open"])
            if verdes >= 6:
                return ResultadoAgente(self.nome, Voto.CALL, 0.45, "poucosswings_tendalta")
            elif verdes <= 4:
                return ResultadoAgente(self.nome, Voto.PUT, 0.45, "poucosswings_tendbaixa")
            return ResultadoAgente(self.nome, Voto.NEUTRO, 0.3, "poucos_swings")

        # Analisa estrutura
        last_high, prev_high = highs[-1], highs[-2]
        last_low, prev_low = lows[-1], lows[-2]

        is_lh = last_high[1] < prev_high[1]  # Lower High
        is_ll = last_low[1] < prev_low[1]    # Lower Low
        is_hh = last_high[1] > prev_high[1]  # Higher High
        is_hl = last_low[1] > prev_low[1]    # Higher Low

        current_price = df_m1.iloc[-1]["close"]

        # DOWNTREND: LH + LL
        if is_lh and is_ll:
            # Verifica se esta em pullback (subiu do ultimo fundo)
            if last_low[0] < len(df_m1) - 1:
                move_down = prev_high[1] - last_low[1]
                retrace_up = current_price - last_low[1]
                if move_down > 0:
                    pullback_pct = retrace_up / move_down
                    if 0.20 <= pullback_pct <= 0.70:
                        conf = min(0.85, 0.55 + pullback_pct * 0.3)
                        return ResultadoAgente(self.nome, Voto.PUT, conf,
                                             f"LH_LL_pullback_{pullback_pct*100:.0f}%")

            return ResultadoAgente(self.nome, Voto.PUT, 0.55, "LH_LL_downtrend")

        # UPTREND: HH + HL
        if is_hh and is_hl:
            # Verifica se esta em pullback (caiu do ultimo topo)
            if last_high[0] < len(df_m1) - 1:
                move_up = last_high[1] - prev_low[1]
                retrace_down = last_high[1] - current_price
                if move_up > 0:
                    pullback_pct = retrace_down / move_up
                    if 0.20 <= pullback_pct <= 0.70:
                        conf = min(0.85, 0.55 + pullback_pct * 0.3)
                        return ResultadoAgente(self.nome, Voto.CALL, conf,
                                             f"HH_HL_pullback_{pullback_pct*100:.0f}%")

            return ResultadoAgente(self.nome, Voto.CALL, 0.55, "HH_HL_uptrend")

        # Estrutura mista - vota baseado na ultima vela
        last = df_m1.iloc[-1]
        if last["close"] > last["open"]:
            return ResultadoAgente(self.nome, Voto.CALL, 0.40, "estrutura_mista_vela_verde")
        elif last["close"] < last["open"]:
            return ResultadoAgente(self.nome, Voto.PUT, 0.40, "estrutura_mista_vela_vermelha")

        return ResultadoAgente(self.nome, Voto.NEUTRO, 0.3, "estrutura_indefinida")


# ============================================================================
#                         AGENTE 6: PERNADA B (ELLIOTT WAVE)
# ============================================================================

class AgentePernada(AgenteBase):
    """
    Agente que analisa Pernada B de Elliott Wave.

    Logica:
    1. Detecta Pernada A (impulso): movimento forte de pelo menos 1.2 ATR
    2. Detecta Pullback: correcao de 30-62% (zona Fibonacci)
    3. Entrada na direcao da Pernada A (continuacao de tendencia)

    Exemplo:
    - Pernada A = queda forte (PUT)
    - Pullback = subida de 38-50%
    - Entrada = PUT (continua a queda)
    """

    def __init__(self):
        super().__init__("Pernada")
        # Parametros da Pernada B
        self.impulso_min_atr = 1.0      # Impulso minimo em ATR
        self.pullback_min = 2            # Velas minimas no pullback
        self.pullback_max = 6            # Velas maximas no pullback
        self.retr_min = 0.25             # Retracao minima (25%)
        self.retr_max = 0.65             # Retracao maxima (65%)
        self.min_eff = 0.55              # Eficiencia minima do impulso

    def _leg_efficiency(self, df: pd.DataFrame) -> float:
        """Calcula eficiencia da pernada (net/gross)."""
        if len(df) < 2:
            return 0.0
        start = df.iloc[0]["open"]
        end = df.iloc[-1]["close"]
        net = abs(end - start)
        gross = sum(abs(r["close"] - r["open"]) for _, r in df.iterrows())
        return net / max(gross, 1e-9)

    def _candle_dir(self, row) -> int:
        """Retorna direcao da vela: 1=alta, -1=baixa, 0=doji"""
        if row["close"] > row["open"]:
            return 1
        elif row["close"] < row["open"]:
            return -1
        return 0

    def analisar(self, df_m1: pd.DataFrame, atr_val: float, **kwargs) -> ResultadoAgente:
        if len(df_m1) < 30:
            return ResultadoAgente(self.nome, Voto.NEUTRO, 0.3, "poucas_velas")

        atr_safe = max(atr_val, 1e-9)
        best_result = None
        best_score = 0.0

        # Testa diferentes tamanhos de pullback
        for pb_len in range(self.pullback_min, self.pullback_max + 1):
            if len(df_m1) < pb_len + 10:
                continue

            # Pullback = ultimas pb_len velas (antes da vela atual)
            pb = df_m1.iloc[-(pb_len + 1):-1]
            if len(pb) != pb_len:
                continue

            # Testa diferentes tamanhos de impulso
            for imp_len in range(3, 12):
                if len(df_m1) < pb_len + 1 + imp_len:
                    continue

                # Impulso = velas antes do pullback
                imp = df_m1.iloc[-(pb_len + 1 + imp_len):-(pb_len + 1)]
                if len(imp) != imp_len:
                    continue

                # Calcula tamanho do impulso
                top = float(imp["high"].max())
                bot = float(imp["low"].min())
                size_A = top - bot

                # Impulso muito pequeno
                if size_A < self.impulso_min_atr * atr_safe:
                    continue

                # Direcao do impulso
                start = float(imp["open"].iloc[0])
                end = float(imp["close"].iloc[-1])
                move = end - start

                if abs(move) < size_A * 0.3:  # Movimento muito fraco
                    continue

                dir_impulso = "PUT" if move < 0 else "CALL"

                # Eficiencia do impulso
                eff = self._leg_efficiency(imp)
                if eff < self.min_eff:
                    continue

                # Conta velas contra no pullback
                contra = 0
                for _, r in pb.iterrows():
                    d = self._candle_dir(r)
                    if dir_impulso == "PUT" and d == 1:  # Pullback de alta
                        contra += 1
                    if dir_impulso == "CALL" and d == -1:  # Pullback de baixa
                        contra += 1

                # Pullback deve ter pelo menos 40% de velas contra
                if contra < max(1, int(pb_len * 0.40)):
                    continue

                # Calcula retracao
                if dir_impulso == "PUT":
                    pb_high = float(pb["high"].max())
                    retr = (pb_high - bot) / max(size_A, 1e-9)
                else:
                    pb_low = float(pb["low"].min())
                    retr = (top - pb_low) / max(size_A, 1e-9)

                # Retracao fora da zona ideal
                if retr < self.retr_min or retr > self.retr_max:
                    continue

                # Calcula score
                score = 0.5 + (eff * 0.2) + ((1 - abs(retr - 0.38)) * 0.2)

                if score > best_score:
                    best_score = score
                    best_result = {
                        "dir": dir_impulso,
                        "eff": eff,
                        "retr": retr,
                        "pb_len": pb_len,
                        "imp_len": imp_len
                    }

        if best_result:
            conf = min(0.85, best_score)
            motivo = f"pernada_{best_result['dir']}_retr={best_result['retr']*100:.0f}%_eff={best_result['eff']:.2f}"

            if best_result["dir"] == "CALL":
                return ResultadoAgente(self.nome, Voto.CALL, conf, motivo)
            else:
                return ResultadoAgente(self.nome, Voto.PUT, conf, motivo)

        # Sem pernada valida - vota baseado na tendencia simples
        ultimas = df_m1.tail(10)
        verdes = sum(1 for _, r in ultimas.iterrows() if r["close"] > r["open"])
        if verdes >= 6:
            return ResultadoAgente(self.nome, Voto.CALL, 0.40, "sem_pernada_tendalta")
        elif verdes <= 4:
            return ResultadoAgente(self.nome, Voto.PUT, 0.40, "sem_pernada_tendbaixa")

        return ResultadoAgente(self.nome, Voto.NEUTRO, 0.3, "sem_pernada_valida")


# ============================================================================
#                         SISTEMA DE CONSENSO
# ============================================================================

class SistemaConsenso:
    """
    Sistema que coordena os 6 agentes e decide se entra ou nao.

    Agentes:
    1. S/R - Suporte e Resistencia (com contexto de tendencia)
    2. Tendencia - Tendencia local (ultimas 7 velas)
    3. Padroes - Padroes de velas (pinbar, engolfo, etc)
    4. CNN - Rede neural
    5. Estrutura - Topos/fundos + pullbacks (LH/HL)
    6. Pernada - Elliott Wave Pernada B

    REGRA: So entra se 5+ agentes votarem na MESMA direcao.
    """

    def __init__(self, trading_cnn=None):
        self.agente_sr = AgenteSR()
        self.agente_tendencia = AgenteTendencia(janela=7)
        self.agente_padroes = AgentePadroes()
        self.agente_cnn = AgenteCNN(trading_cnn)
        self.agente_estrutura = AgenteEstrutura(swing_lookback=5)
        self.agente_pernada = AgentePernada()

        self.agentes = [
            self.agente_sr,
            self.agente_tendencia,
            self.agente_padroes,
            self.agente_cnn,
            self.agente_estrutura,
            self.agente_pernada
        ]

    def set_cnn_model(self, trading_cnn):
        """Atualiza o modelo CNN."""
        self.agente_cnn.set_model(trading_cnn)

    def analisar(self, df_m1: pd.DataFrame, atr_val: float, sr_levels: Dict = None) -> Dict[str, Any]:
        """
        Analisa com todos os agentes e retorna decisao de consenso.

        Args:
            df_m1: DataFrame com candles M1
            atr_val: Valor do ATR
            sr_levels: Dict com "resistencias" e "suportes"

        Returns:
            Dict com:
            - trade: bool (True se deve entrar)
            - direcao: "CALL", "PUT" ou "NO_TRADE"
            - confianca_media: float
            - votos: lista de ResultadoAgente
            - motivo: string explicando decisao
        """
        votos: List[ResultadoAgente] = []

        # Coleta votos de todos os agentes
        for agente in self.agentes:
            if isinstance(agente, AgenteSR):
                resultado = agente.analisar(df_m1, atr_val, sr_levels=sr_levels)
            else:
                resultado = agente.analisar(df_m1, atr_val)
            votos.append(resultado)

        # Conta votos por direcao
        votos_call = [v for v in votos if v.voto == Voto.CALL]
        votos_put = [v for v in votos if v.voto == Voto.PUT]
        votos_neutro = [v for v in votos if v.voto == Voto.NEUTRO]

        # Monta resumo dos votos
        resumo_votos = " | ".join([f"{v.nome}:{v.voto.value}({v.confianca:.2f})" for v in votos])

        # REGRA DE CONSENSO (6 agentes):
        # - 5+ agentes na mesma direcao = ENTRA
        # - Qualquer conflito CALL vs PUT = NAO ENTRA
        # - Menos de 5 = NAO ENTRA

        # Verifica se ha conflito (CALL e PUT ao mesmo tempo)
        if len(votos_call) > 0 and len(votos_put) > 0:
            return {
                "trade": False,
                "direcao": "NO_TRADE",
                "confianca_media": 0.0,
                "votos": votos,
                "motivo": f"CONFLITO: {len(votos_call)} CALL vs {len(votos_put)} PUT | {resumo_votos}"
            }

        # CONSENSO CALL: Pelo menos 5 agentes votam CALL
        if len(votos_call) >= 5:
            conf_media = sum(v.confianca for v in votos_call) / len(votos_call)
            return {
                "trade": True,
                "direcao": "CALL",
                "confianca_media": conf_media,
                "votos": votos,
                "motivo": f"CONSENSO CALL ({len(votos_call)}/6) | {resumo_votos}"
            }

        # CONSENSO PUT: Pelo menos 5 agentes votam PUT
        if len(votos_put) >= 5:
            conf_media = sum(v.confianca for v in votos_put) / len(votos_put)
            return {
                "trade": True,
                "direcao": "PUT",
                "confianca_media": conf_media,
                "votos": votos,
                "motivo": f"CONSENSO PUT ({len(votos_put)}/6) | {resumo_votos}"
            }

        # Sem consenso suficiente (precisa de 5+)
        return {
            "trade": False,
            "direcao": "NO_TRADE",
            "confianca_media": 0.0,
            "votos": votos,
            "motivo": f"SEM CONSENSO: CALL={len(votos_call)} PUT={len(votos_put)} NEUTRO={len(votos_neutro)} | {resumo_votos}"
        }


# ============================================================================
#                         INSTANCIA GLOBAL
# ============================================================================

_consenso_instance: Optional[SistemaConsenso] = None


def get_consenso(trading_cnn=None) -> SistemaConsenso:
    """Retorna instancia global do sistema de consenso."""
    global _consenso_instance
    if _consenso_instance is None:
        _consenso_instance = SistemaConsenso(trading_cnn)
    elif trading_cnn is not None:
        _consenso_instance.set_cnn_model(trading_cnn)
    return _consenso_instance


# ============================================================================
#                         TESTE LOCAL
# ============================================================================

if __name__ == "__main__":
    print("=== Teste do Sistema de Consenso Multi-Agente ===\n")

    # Cria dados de teste
    import random

    # Simula DataFrame
    data = {
        "open": [100 + random.uniform(-1, 1) for _ in range(100)],
        "high": [101 + random.uniform(-1, 1) for _ in range(100)],
        "low": [99 + random.uniform(-1, 1) for _ in range(100)],
        "close": [100.5 + random.uniform(-1, 1) for _ in range(100)],
    }
    df = pd.DataFrame(data)

    # Cria sistema
    consenso = get_consenso()

    # Testa analise
    sr_levels = {
        "suportes": [(99.5, 5), (99.0, 3)],
        "resistencias": [(101.0, 4), (101.5, 2)]
    }

    resultado = consenso.analisar(df, atr_val=0.5, sr_levels=sr_levels)

    print(f"Trade: {resultado['trade']}")
    print(f"Direcao: {resultado['direcao']}")
    print(f"Confianca: {resultado['confianca_media']:.2f}")
    print(f"Motivo: {resultado['motivo']}")
    print("\nVotos individuais:")
    for v in resultado['votos']:
        print(f"  - {v.nome}: {v.voto.value} (conf={v.confianca:.2f}) - {v.motivo}")
