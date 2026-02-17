"""
Regime Filter - Filtros Pre-Modelo

Objetivo: Bloquear mercados ruins ANTES de passar pelo modelo CNN.
Se qualquer filtro falhar, nao opera (NO TRADE).

Filtros:
1. Volatilidade muito baixa (mercado morto)
2. Volatilidade muito alta (serrilhado)
3. Alternancia excessiva (chop)
4. Pavios dominantes repetidos
5. Payout abaixo do minimo
"""

from typing import Tuple, Dict, Optional
import pandas as pd
import numpy as np

# ============================================================================
#                         PARAMETROS DO REGIME FILTER
# ============================================================================

# Volatilidade (ATR normalizado pelo preco)
MIN_ATR_RATIO = 0.0003      # Volatilidade minima (0.03% do preco)
MAX_ATR_RATIO = 0.0080      # Volatilidade maxima (0.80% do preco)

# Choppiness (mercado lateral/serrilhado)
CHOP_LOOKBACK = 20          # Ultimas 20 velas para avaliar chop
CHOP_MAX_FLIPS = 0.75       # Max 75% de flips de cor (verde/vermelho)
CHOP_MIN_EFFICIENCY = 0.15  # Min 15% de eficiencia direcional

# Pavios dominantes
WICK_LOOKBACK = 5           # Ultimas 5 velas para avaliar pavios
WICK_MAX_RATIO = 0.65       # Max 65% do range como pavio
WICK_MAX_DOMINANT = 4       # Max 4 de 5 velas com pavio dominante

# Payout minimo
MIN_PAYOUT = 70             # Payout minimo 70%

# Tendencia M5
M5_MIN_STRENGTH = 0.30      # Forca minima da tendencia no M5

# ============================================================================
#                         CLASSE REGIME FILTER
# ============================================================================

class RegimeFilter:
    """
    Filtro de regime de mercado.
    Bloqueia operacoes em condicoes desfavoraveis ANTES do modelo.
    """

    def __init__(
        self,
        min_atr_ratio: float = MIN_ATR_RATIO,
        max_atr_ratio: float = MAX_ATR_RATIO,
        chop_lookback: int = CHOP_LOOKBACK,
        chop_max_flips: float = CHOP_MAX_FLIPS,
        chop_min_efficiency: float = CHOP_MIN_EFFICIENCY,
        wick_lookback: int = WICK_LOOKBACK,
        wick_max_ratio: float = WICK_MAX_RATIO,
        wick_max_dominant: int = WICK_MAX_DOMINANT,
        min_payout: int = MIN_PAYOUT,
        m5_min_strength: float = M5_MIN_STRENGTH
    ):
        self.min_atr_ratio = min_atr_ratio
        self.max_atr_ratio = max_atr_ratio
        self.chop_lookback = chop_lookback
        self.chop_max_flips = chop_max_flips
        self.chop_min_efficiency = chop_min_efficiency
        self.wick_lookback = wick_lookback
        self.wick_max_ratio = wick_max_ratio
        self.wick_max_dominant = wick_max_dominant
        self.min_payout = min_payout
        self.m5_min_strength = m5_min_strength

    def should_block(
        self,
        df_m1: pd.DataFrame,
        df_m5: Optional[pd.DataFrame] = None,
        atr_val: Optional[float] = None,
        payout: int = 80
    ) -> Tuple[bool, str, Dict]:
        """
        Verifica se deve bloquear a operacao.

        Args:
            df_m1: DataFrame com candles M1
            df_m5: DataFrame com candles M5 (opcional)
            atr_val: Valor ATR pre-calculado (opcional)
            payout: Payout atual do ativo

        Returns:
            Tuple[bool, str, Dict]: (bloqueado, motivo, detalhes)
        """
        details = {}

        # Calcula ATR se nao fornecido
        if atr_val is None:
            atr_val = self._calculate_atr(df_m1)

        # Preco atual para normalizar ATR
        current_price = df_m1['close'].iloc[-1]
        atr_ratio = atr_val / current_price if current_price > 0 else 0
        details['atr_ratio'] = atr_ratio

        # 1. Volatilidade muito baixa (mercado morto)
        if atr_ratio < self.min_atr_ratio:
            details['reason_detail'] = f"ATR ratio {atr_ratio:.5f} < {self.min_atr_ratio}"
            return True, "LOW_VOLATILITY", details

        # 2. Volatilidade muito alta (serrilhado)
        if atr_ratio > self.max_atr_ratio:
            details['reason_detail'] = f"ATR ratio {atr_ratio:.5f} > {self.max_atr_ratio}"
            return True, "HIGH_VOLATILITY", details

        # 3. Alternancia excessiva (chop)
        chop_blocked, chop_info = self._check_choppiness(df_m1)
        details['chop_info'] = chop_info
        if chop_blocked:
            details['reason_detail'] = f"Flip rate {chop_info['flip_rate']:.2f}, efficiency {chop_info['efficiency']:.2f}"
            return True, "CHOPPY_MARKET", details

        # 4. Pavios dominantes repetidos
        wick_blocked, wick_info = self._check_dominant_wicks(df_m1)
        details['wick_info'] = wick_info
        if wick_blocked:
            details['reason_detail'] = f"Dominant wicks: {wick_info['dominant_count']}/{self.wick_lookback}"
            return True, "DOMINANT_WICKS", details

        # 5. Payout abaixo do minimo
        details['payout'] = payout
        if payout < self.min_payout:
            details['reason_detail'] = f"Payout {payout}% < {self.min_payout}%"
            return True, "LOW_PAYOUT", details

        # 6. Tendencia M5 (se fornecido)
        if df_m5 is not None and len(df_m5) >= 20:
            m5_blocked, m5_info = self._check_m5_trend(df_m5)
            details['m5_info'] = m5_info
            if m5_blocked:
                details['reason_detail'] = f"M5 trend: {m5_info['direction']}, strength: {m5_info['strength']:.2f}"
                return True, "WEAK_M5_TREND", details

        return False, "OK", details

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calcula ATR (Average True Range)."""
        if len(df) < period:
            return 0.0

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr = np.maximum(
            high[-period:] - low[-period:],
            np.maximum(
                np.abs(high[-period:] - np.roll(close, 1)[-period:]),
                np.abs(low[-period:] - np.roll(close, 1)[-period:])
            )
        )
        return float(np.mean(tr[1:]))  # Skip first (invalid due to roll)

    def _check_choppiness(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Verifica se o mercado esta choppy (lateral/serrilhado).

        Criterios:
        - Taxa de flips de cor (verde/vermelho) > threshold
        - Eficiencia direcional < threshold
        """
        if len(df) < self.chop_lookback:
            return False, {'flip_rate': 0, 'efficiency': 1}

        recent = df.tail(self.chop_lookback)

        # Calcula direcao de cada vela
        directions = (recent['close'] > recent['open']).astype(int).values

        # Conta flips (mudancas de direcao)
        flips = np.sum(np.abs(np.diff(directions)))
        max_flips = len(directions) - 1
        flip_rate = flips / max_flips if max_flips > 0 else 0

        # Calcula eficiencia direcional
        # Net move / Gross move
        net_move = abs(recent['close'].iloc[-1] - recent['open'].iloc[0])
        gross_move = (recent['high'] - recent['low']).sum()
        efficiency = net_move / gross_move if gross_move > 0 else 0

        info = {
            'flip_rate': flip_rate,
            'efficiency': efficiency,
            'flips': flips,
            'max_flips': max_flips
        }

        # Bloqueia se muitos flips OU baixa eficiencia
        blocked = flip_rate > self.chop_max_flips or efficiency < self.chop_min_efficiency

        return blocked, info

    def _check_dominant_wicks(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Verifica se ha pavios dominantes repetidos.

        Pavio dominante = pavio > 60% do range total da vela
        """
        if len(df) < self.wick_lookback:
            return False, {'dominant_count': 0, 'wick_ratios': []}

        recent = df.tail(self.wick_lookback)

        wick_ratios = []
        dominant_count = 0

        for i in range(len(recent)):
            row = recent.iloc[i]
            candle_range = row['high'] - row['low']

            if candle_range <= 0:
                wick_ratios.append(0)
                continue

            body_top = max(row['open'], row['close'])
            body_bottom = min(row['open'], row['close'])

            upper_wick = row['high'] - body_top
            lower_wick = body_bottom - row['low']
            total_wick = upper_wick + lower_wick

            wick_ratio = total_wick / candle_range
            wick_ratios.append(wick_ratio)

            if wick_ratio > self.wick_max_ratio:
                dominant_count += 1

        info = {
            'dominant_count': dominant_count,
            'wick_ratios': wick_ratios,
            'avg_wick_ratio': np.mean(wick_ratios) if wick_ratios else 0
        }

        blocked = dominant_count >= self.wick_max_dominant

        return blocked, info

    def _check_m5_trend(self, df_m5: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Verifica a forca da tendencia no timeframe M5.

        Usa multiplos indicadores: EMA, slope, estrutura de topos/fundos.
        """
        if len(df_m5) < 50:
            return False, {'direction': 'NEUTRAL', 'strength': 0.5}

        close = df_m5['close'].values
        high = df_m5['high'].values
        low = df_m5['low'].values

        # Calcula EMAs
        ema10 = self._ema(close, 10)
        ema20 = self._ema(close, 20)
        ema50 = self._ema(close, 50)

        # === METODO 1: EMA alignment ===
        ema_bullish = ema10 > ema20 > ema50
        ema_bearish = ema10 < ema20 < ema50

        # === METODO 2: Slope (inclinacao das ultimas 10 velas) ===
        slope_period = min(10, len(close) - 1)
        price_change = close[-1] - close[-slope_period - 1]
        avg_price = np.mean(close[-slope_period:])
        slope_pct = (price_change / avg_price) * 100 if avg_price > 0 else 0

        slope_bullish = slope_pct > 0.01  # Subindo > 0.01% (mais sensível)
        slope_bearish = slope_pct < -0.01  # Caindo > 0.01% (mais sensível)

        # === METODO 3: Estrutura (HH/HL vs LH/LL) ===
        structure = self._detect_structure(df_m5)
        struct_bullish = structure == 'HH_HL'
        struct_bearish = structure == 'LH_LL'

        # === METODO 4: Close vs Open das ultimas N velas ===
        last_10 = df_m5.tail(10)
        bullish_candles = sum(1 for _, r in last_10.iterrows() if r['close'] > r['open'])
        bearish_candles = 10 - bullish_candles
        candle_bullish = bullish_candles >= 5  # 5 de 10 é suficiente
        candle_bearish = bearish_candles >= 5  # 5 de 10 é suficiente

        # === DECISAO COMBINADA ===
        # Conta sinais de cada direcao
        bull_signals = sum([ema_bullish, slope_bullish, struct_bullish, candle_bullish])
        bear_signals = sum([ema_bearish, slope_bearish, struct_bearish, candle_bearish])

        # Determina direcao - MAIS FLEXÍVEL
        # 1+ sinal forte (slope OU estrutura) = direção detectada
        # OU 2+ sinais fracos = direção detectada
        strong_bull = slope_bullish or struct_bullish
        strong_bear = slope_bearish or struct_bearish

        if (bull_signals >= 2 or strong_bull) and bull_signals > bear_signals:
            direction = 'BULLISH'
            strength = min(1.0, (bull_signals + 1) / 5.0)  # Bonus por ter direção
        elif (bear_signals >= 2 or strong_bear) and bear_signals > bull_signals:
            direction = 'BEARISH'
            strength = min(1.0, (bear_signals + 1) / 5.0)  # Bonus por ter direção
        elif bull_signals > bear_signals and bull_signals >= 1:
            # Pelo menos 1 sinal bullish e mais que bearish
            direction = 'BULLISH'
            strength = 0.25  # Força baixa mas direção definida
        elif bear_signals > bull_signals and bear_signals >= 1:
            # Pelo menos 1 sinal bearish e mais que bullish
            direction = 'BEARISH'
            strength = 0.25  # Força baixa mas direção definida
        else:
            direction = 'NEUTRAL'
            strength = 0.0

        info = {
            'direction': direction,
            'strength': strength,
            'ema10': ema10,
            'ema20': ema20,
            'ema50': ema50,
            'slope_pct': slope_pct,
            'structure': structure,
            'bull_signals': bull_signals,
            'bear_signals': bear_signals,
            'details': f"EMA:{ema_bullish or ema_bearish},Slope:{slope_bullish or slope_bearish},Struct:{structure},Candles:{bullish_candles}B/{bearish_candles}S"
        }

        # Bloqueia se tendencia muito fraca ou neutra
        blocked = direction == 'NEUTRAL' or strength < self.m5_min_strength

        return blocked, info

    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calcula EMA (Exponential Moving Average)."""
        if len(data) < period:
            return float(data[-1]) if len(data) > 0 else 0.0

        multiplier = 2 / (period + 1)
        ema = data[-period]

        for price in data[-period + 1:]:
            ema = (price - ema) * multiplier + ema

        return float(ema)

    def _detect_structure(self, df: pd.DataFrame) -> str:
        """
        Detecta estrutura de mercado (HH/HL, LH/LL, ou RANGE).

        HH/HL = Higher Highs, Higher Lows (tendencia de alta)
        LH/LL = Lower Highs, Lower Lows (tendencia de baixa)
        """
        if len(df) < 20:
            return 'RANGE'

        recent = df.tail(20)

        # Encontra pivots (maximos e minimos locais)
        highs = recent['high'].values
        lows = recent['low'].values

        # Simplificado: compara primeira e segunda metade
        first_half_high = np.max(highs[:10])
        second_half_high = np.max(highs[10:])
        first_half_low = np.min(lows[:10])
        second_half_low = np.min(lows[10:])

        hh = second_half_high > first_half_high
        hl = second_half_low > first_half_low
        lh = second_half_high < first_half_high
        ll = second_half_low < first_half_low

        if hh and hl:
            return 'HH_HL'  # Bullish structure
        elif lh and ll:
            return 'LH_LL'  # Bearish structure
        else:
            return 'RANGE'

    def get_m5_direction(self, df_m5: pd.DataFrame) -> Tuple[str, float]:
        """
        Retorna a direcao e forca da tendencia M5.

        Usado para validar se a operacao M1 esta na direcao correta.

        Returns:
            Tuple[str, float]: (direcao, forca)
            direcao: 'BULLISH', 'BEARISH', 'NEUTRAL'
            forca: 0.0 a 1.0
        """
        if df_m5 is None or len(df_m5) < 50:
            return 'NEUTRAL', 0.0

        _, info = self._check_m5_trend(df_m5)
        return info['direction'], info['strength']


# ============================================================================
#                         FUNCOES AUXILIARES
# ============================================================================

def create_regime_filter(**kwargs) -> RegimeFilter:
    """Factory function para criar RegimeFilter com parametros customizados."""
    return RegimeFilter(**kwargs)


# ============================================================================
#                         TESTE LOCAL
# ============================================================================

if __name__ == "__main__":
    # Teste basico
    import numpy as np

    # Cria dados de teste
    np.random.seed(42)
    n = 100

    # Simula candles
    close = 1.0 + np.cumsum(np.random.randn(n) * 0.001)
    open_prices = close - np.random.randn(n) * 0.0005
    high = np.maximum(close, open_prices) + np.abs(np.random.randn(n) * 0.0003)
    low = np.minimum(close, open_prices) - np.abs(np.random.randn(n) * 0.0003)

    df_test = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close
    })

    # Testa o filtro
    rf = RegimeFilter()
    blocked, reason, details = rf.should_block(df_test, payout=80)

    print(f"Bloqueado: {blocked}")
    print(f"Motivo: {reason}")
    print(f"Detalhes: {details}")
