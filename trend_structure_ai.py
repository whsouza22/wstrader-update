"""
Trend Structure AI - Detecta estrutura de tendencia com topos/fundos

Logica:
- DOWNTREND: Topos mais baixos (LH) + Fundos mais baixos (LL)
- UPTREND: Fundos mais altos (HL) + Topos mais altos (HH)
- Pullback na tendencia = oportunidade de entrada NA DIRECAO da tendencia

Exemplo do grafico do usuario:
- Topo 1 em X, Topo 2 mais baixo que Topo 1 = LH (Lower High)
- Preco faz pullback (sobe um pouco)
- Continua a queda = entrada PUT no pullback
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TrendDirection(Enum):
    UPTREND = "UPTREND"      # HH + HL
    DOWNTREND = "DOWNTREND"  # LH + LL
    SIDEWAYS = "SIDEWAYS"    # Sem estrutura clara


@dataclass
class SwingPoint:
    """Ponto de swing (topo ou fundo)"""
    index: int
    price: float
    is_high: bool  # True = topo, False = fundo


@dataclass
class TrendStructure:
    """Estrutura de tendencia detectada"""
    direction: TrendDirection
    strength: float  # 0.0 a 1.0
    last_swing_high: Optional[SwingPoint]
    last_swing_low: Optional[SwingPoint]
    prev_swing_high: Optional[SwingPoint]
    prev_swing_low: Optional[SwingPoint]
    is_pullback: bool
    pullback_depth: float  # % do movimento que retracou
    signal: str  # "CALL", "PUT", "WAIT"
    reason: str


class TrendStructureAI:
    """
    IA que detecta estrutura de tendencia e pullbacks.

    Filosofia:
    - Identifica topos e fundos significativos
    - Compara com anteriores para determinar tendencia
    - Detecta pullbacks como oportunidades
    - Opera NA DIRECAO da tendencia maior
    """

    def __init__(self, swing_lookback: int = 5, min_swing_size: float = 0.0003):
        """
        Args:
            swing_lookback: Velas para considerar um swing point
            min_swing_size: Tamanho minimo do swing em % do preco
        """
        self.swing_lookback = swing_lookback
        self.min_swing_size = min_swing_size

    def find_swing_points(self, df: pd.DataFrame, n_points: int = 4) -> List[SwingPoint]:
        """
        Encontra os ultimos N pontos de swing (topos e fundos).

        Um topo e uma vela com high maior que as `swing_lookback` velas antes e depois.
        Um fundo e uma vela com low menor que as `swing_lookback` velas antes e depois.
        """
        if len(df) < self.swing_lookback * 2 + 1:
            return []

        swings = []
        lb = self.swing_lookback

        # Percorre o dataframe procurando swings
        for i in range(lb, len(df) - lb):
            high = df.iloc[i]["high"]
            low = df.iloc[i]["low"]

            # Verifica se e um topo (swing high)
            is_swing_high = True
            for j in range(i - lb, i + lb + 1):
                if j != i and df.iloc[j]["high"] >= high:
                    is_swing_high = False
                    break

            if is_swing_high:
                swings.append(SwingPoint(i, high, True))
                continue

            # Verifica se e um fundo (swing low)
            is_swing_low = True
            for j in range(i - lb, i + lb + 1):
                if j != i and df.iloc[j]["low"] <= low:
                    is_swing_low = False
                    break

            if is_swing_low:
                swings.append(SwingPoint(i, low, False))

        # Retorna os ultimos N pontos
        return swings[-n_points:] if len(swings) >= n_points else swings

    def analyze_structure(self, df: pd.DataFrame) -> TrendStructure:
        """
        Analisa a estrutura de tendencia atual.

        Returns:
            TrendStructure com direcao, forca e sinal
        """
        if len(df) < 30:
            return TrendStructure(
                direction=TrendDirection.SIDEWAYS,
                strength=0.0,
                last_swing_high=None,
                last_swing_low=None,
                prev_swing_high=None,
                prev_swing_low=None,
                is_pullback=False,
                pullback_depth=0.0,
                signal="WAIT",
                reason="dados_insuficientes"
            )

        # Encontra swing points
        swings = self.find_swing_points(df, n_points=6)

        if len(swings) < 4:
            return TrendStructure(
                direction=TrendDirection.SIDEWAYS,
                strength=0.0,
                last_swing_high=None,
                last_swing_low=None,
                prev_swing_high=None,
                prev_swing_low=None,
                is_pullback=False,
                pullback_depth=0.0,
                signal="WAIT",
                reason="poucos_swings"
            )

        # Separa topos e fundos
        highs = [s for s in swings if s.is_high]
        lows = [s for s in swings if not s.is_high]

        if len(highs) < 2 or len(lows) < 2:
            return TrendStructure(
                direction=TrendDirection.SIDEWAYS,
                strength=0.0,
                last_swing_high=highs[-1] if highs else None,
                last_swing_low=lows[-1] if lows else None,
                prev_swing_high=highs[-2] if len(highs) > 1 else None,
                prev_swing_low=lows[-2] if len(lows) > 1 else None,
                is_pullback=False,
                pullback_depth=0.0,
                signal="WAIT",
                reason="estrutura_incompleta"
            )

        # Pega os 2 ultimos de cada
        last_high, prev_high = highs[-1], highs[-2]
        last_low, prev_low = lows[-1], lows[-2]

        # Analisa estrutura
        # DOWNTREND: LH (Lower High) + LL (Lower Low)
        is_lh = last_high.price < prev_high.price
        is_ll = last_low.price < prev_low.price

        # UPTREND: HH (Higher High) + HL (Higher Low)
        is_hh = last_high.price > prev_high.price
        is_hl = last_low.price > prev_low.price

        # Preco atual
        current_price = df.iloc[-1]["close"]

        # Determina direcao
        if is_lh and is_ll:
            direction = TrendDirection.DOWNTREND
            strength = self._calc_strength(prev_high.price, last_high.price, prev_low.price, last_low.price)

            # Verifica pullback (preco subiu do ultimo fundo)
            pullback_depth = 0.0
            is_pullback = False

            if last_low.index < len(df) - 1:  # Fundo nao e a ultima vela
                move_down = prev_high.price - last_low.price
                retrace_up = current_price - last_low.price

                if move_down > 0:
                    pullback_depth = retrace_up / move_down
                    # Pullback entre 30% e 70% e ideal
                    is_pullback = 0.20 <= pullback_depth <= 0.75

            if is_pullback:
                signal = "PUT"
                reason = f"downtrend_pullback_{pullback_depth*100:.0f}%"
            else:
                signal = "WAIT"
                reason = f"downtrend_sem_pullback"

        elif is_hh and is_hl:
            direction = TrendDirection.UPTREND
            strength = self._calc_strength(prev_low.price, last_low.price, prev_high.price, last_high.price)

            # Verifica pullback (preco caiu do ultimo topo)
            pullback_depth = 0.0
            is_pullback = False

            if last_high.index < len(df) - 1:  # Topo nao e a ultima vela
                move_up = last_high.price - prev_low.price
                retrace_down = last_high.price - current_price

                if move_up > 0:
                    pullback_depth = retrace_down / move_up
                    is_pullback = 0.20 <= pullback_depth <= 0.75

            if is_pullback:
                signal = "CALL"
                reason = f"uptrend_pullback_{pullback_depth*100:.0f}%"
            else:
                signal = "WAIT"
                reason = f"uptrend_sem_pullback"
        else:
            direction = TrendDirection.SIDEWAYS
            strength = 0.3
            is_pullback = False
            pullback_depth = 0.0
            signal = "WAIT"
            reason = "sem_estrutura_clara"

        return TrendStructure(
            direction=direction,
            strength=strength,
            last_swing_high=last_high,
            last_swing_low=last_low,
            prev_swing_high=prev_high,
            prev_swing_low=prev_low,
            is_pullback=is_pullback,
            pullback_depth=pullback_depth,
            signal=signal,
            reason=reason
        )

    def _calc_strength(self, p1: float, p2: float, p3: float, p4: float) -> float:
        """Calcula forca da tendencia baseado na consistencia dos swings."""
        try:
            # Quanto maior a diferenca entre swings, mais forte a tendencia
            diff1 = abs(p2 - p1) / p1 if p1 > 0 else 0
            diff2 = abs(p4 - p3) / p3 if p3 > 0 else 0

            avg_diff = (diff1 + diff2) / 2

            # Normaliza para 0-1
            strength = min(1.0, avg_diff * 100)  # 1% de diferenca = forca 1.0
            return max(0.3, strength)
        except:
            return 0.5

    def get_entry_signal(self, df: pd.DataFrame) -> Dict:
        """
        Retorna sinal de entrada baseado na estrutura.

        Returns:
            {
                "trade": bool,
                "direction": "CALL" | "PUT",
                "confidence": float,
                "structure": TrendStructure,
                "reason": str
            }
        """
        structure = self.analyze_structure(df)

        if structure.signal == "WAIT":
            return {
                "trade": False,
                "direction": None,
                "confidence": 0.0,
                "structure": structure,
                "reason": structure.reason
            }

        # Confirma com a ultima vela
        last_candle = df.iloc[-1]
        candle_dir = "CALL" if last_candle["close"] > last_candle["open"] else "PUT"

        # Se a vela confirma a direcao, aumenta confianca
        if candle_dir == structure.signal:
            confidence = min(1.0, structure.strength + 0.2)
            reason = f"{structure.reason}_confirmado"
        else:
            # Vela contra - pode ser o pullback ainda em andamento
            confidence = structure.strength * 0.8
            reason = f"{structure.reason}_aguardando_confirmacao"

        return {
            "trade": confidence >= 0.5,
            "direction": structure.signal,
            "confidence": confidence,
            "structure": structure,
            "reason": reason
        }


# ============================================================================
#                         TESTE LOCAL
# ============================================================================

if __name__ == "__main__":
    # Simula dados com tendencia de baixa (como no grafico do usuario)
    np.random.seed(42)

    # Cria tendencia de baixa com pullbacks
    n = 60
    data = []
    price = 1.17500

    for i in range(n):
        # Tendencia geral de baixa
        trend = -0.00010

        # Adiciona pullbacks periodicos
        if i % 15 < 5:  # Pullback a cada 15 velas por 5 velas
            trend = 0.00005

        noise = np.random.normal(0, 0.00020)

        open_p = price
        close_p = price + trend + noise
        high_p = max(open_p, close_p) + abs(np.random.normal(0, 0.00010))
        low_p = min(open_p, close_p) - abs(np.random.normal(0, 0.00010))

        data.append({
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p
        })

        price = close_p

    df = pd.DataFrame(data)

    # Testa a IA
    ai = TrendStructureAI()

    print("=== Analise de Estrutura ===")
    structure = ai.analyze_structure(df)
    print(f"Direcao: {structure.direction.value}")
    print(f"Forca: {structure.strength:.2f}")
    print(f"Pullback: {structure.is_pullback} ({structure.pullback_depth*100:.1f}%)")
    print(f"Sinal: {structure.signal}")
    print(f"Motivo: {structure.reason}")

    print("\n=== Sinal de Entrada ===")
    signal = ai.get_entry_signal(df)
    print(f"Trade: {signal['trade']}")
    print(f"Direcao: {signal['direction']}")
    print(f"Confianca: {signal['confidence']:.2f}")
    print(f"Motivo: {signal['reason']}")
