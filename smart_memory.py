# -*- coding: utf-8 -*-
"""
Smart Memory - Sistema de Memoria Inteligente para Trading
Aprende com wins e losses, aplica peso temporal, bloqueia combinacoes ruins.

Features:
- Persiste razoes especificas de cada trade
- Peso temporal: losses recentes pesam mais (decaimento de 8%/dia)
- Bloqueia combinacoes ativo+padrao+contexto com WR < 35%
- Aprende com wins: reforca padroes vencedores (peso 1.5x)
- Detecta sequencias perdedoras e bloqueia automaticamente
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ===================== CONFIGURACAO =====================
MEMORY_FILE = os.getenv("WS_SMART_MEMORY", "ws_smart_memory.json")
TEMPORAL_DECAY_RATE = float(os.getenv("WS_MEMORY_DECAY", "0.92"))  # 8% decay por dia
RECENT_WINDOW_HOURS = int(os.getenv("WS_MEMORY_RECENT", "24"))  # Ultimas 24h pesam 2x
WIN_REINFORCE_WEIGHT = float(os.getenv("WS_WIN_WEIGHT", "1.5"))  # Wins pesam 1.5x
BLOCK_WR_THRESHOLD = float(os.getenv("WS_BLOCK_WR", "0.35"))  # Bloqueia se WR < 35%
MIN_TRADES_TO_BLOCK = int(os.getenv("WS_MIN_TRADES_BLOCK", "3"))  # Minimo 3 trades para bloquear (aprende rapido!)
LOSING_STREAK_BLOCK = int(os.getenv("WS_LOSING_STREAK", "2"))  # Bloqueia apos 2 losses seguidos (aprende rapido!)


class SmartMemory:
    """
    Sistema de Memoria Inteligente para Trading
    Persiste razoes de wins/losses e bloqueia combinacoes ruins.
    """

    def __init__(self, memory_file: str = None):
        self.memory_file = memory_file or MEMORY_FILE
        self.combinations: Dict[str, Dict] = {}  # ativo_direcao_contexto -> stats
        self.blocked: set = set()  # Combinacoes bloqueadas
        self.recent_trades: List[Dict] = []  # Ultimos 50 trades
        self.last_save = 0
        self._load()

    def _load(self):
        """Carrega memoria do disco"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.combinations = data.get("combinations", {})
                    self.blocked = set(data.get("blocked", []))
                    self.recent_trades = data.get("recent_trades", [])
                    logger.info(f"[MEMORY] Carregado: {len(self.combinations)} combinacoes, {len(self.blocked)} bloqueados")
        except Exception as e:
            logger.warning(f"[MEMORY] Erro ao carregar: {e}")

    def _save(self):
        """Salva memoria no disco (max 1x por minuto)"""
        now = time.time()
        if now - self.last_save < 60:
            return

        try:
            data = {
                "combinations": self.combinations,
                "blocked": list(self.blocked),
                "recent_trades": self.recent_trades[-50:],  # Mantem ultimos 50
                "updated_at": datetime.now().isoformat()
            }
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.last_save = now
        except Exception as e:
            logger.warning(f"[MEMORY] Erro ao salvar: {e}")

    def _make_key(self, ativo: str, direction: str, context: str = "default") -> str:
        """Cria chave unica para a combinacao"""
        return f"{ativo}_{direction}_{context}".upper()

    def _extract_context(self, trade_data: Dict) -> str:
        """Extrai contexto do trade para criar chave mais especifica"""
        context_parts = []

        market_ctx = trade_data.get("market_context", {})

        # Tendencia
        trend = market_ctx.get("trend", "unknown")
        if trend:
            context_parts.append(trend[:4])  # bull, bear, late

        # Volatilidade
        vol = market_ctx.get("volatility", "")
        if vol:
            context_parts.append(vol[:4])  # high, low, norm

        # Consolidacao
        if market_ctx.get("is_consolidating"):
            context_parts.append("cons")

        # Padrao detectado
        padrao = trade_data.get("padrao", "")
        if padrao:
            context_parts.append(padrao[:6])

        return "_".join(context_parts) if context_parts else "default"

    def _extract_reasons(self, trade_data: Dict, is_win: bool) -> List[str]:
        """Extrai razoes do resultado do trade"""
        reasons = []

        market_ctx = trade_data.get("market_context", {})
        entry_quality = trade_data.get("entry_quality", {})

        if is_win:
            # Razoes de sucesso
            if market_ctx.get("trend") == trade_data.get("direction", "").lower():
                reasons.append("tendencia_favor")
            if entry_quality.get("entry_quality") == "strong":
                reasons.append("entrada_forte")
            if entry_quality.get("alignment_ratio", 0) > 0.6:
                reasons.append("alinhamento_bom")
            if not market_ctx.get("is_consolidating"):
                reasons.append("sem_consolidacao")
        else:
            # Razoes de falha
            trend = market_ctx.get("trend", "")
            direction = trade_data.get("direction", "").lower()

            if trend == "bullish" and direction == "put":
                reasons.append("contra_tendencia")
            elif trend == "bearish" and direction == "call":
                reasons.append("contra_tendencia")

            if market_ctx.get("is_consolidating"):
                reasons.append("consolidacao")

            if market_ctx.get("near_resistance") and direction == "call":
                reasons.append("sr_forte")
            if market_ctx.get("near_support") and direction == "put":
                reasons.append("sr_forte")

            if entry_quality.get("entry_quality") == "weak":
                reasons.append("entrada_fraca")

            if entry_quality.get("alignment_ratio", 0) < 0.4:
                reasons.append("desalinhamento")

            if entry_quality.get("momentum_direction") == "wrong":
                reasons.append("momentum_errado")

        return reasons if reasons else ["unknown"]

    def record_trade(self, trade_data: Dict):
        """
        Registra um trade com todas as informacoes

        trade_data deve conter:
        - ativo: str
        - direction: str (CALL/PUT)
        - profit: float (positivo = win, negativo = loss)
        - market_context: Dict (opcional)
        - entry_quality: Dict (opcional)
        - padrao: str (opcional)
        - timestamp: float (opcional)
        """
        ativo = trade_data.get("ativo", "UNKNOWN")
        direction = trade_data.get("direction", "UNKNOWN")
        profit = trade_data.get("profit", 0)
        timestamp = trade_data.get("timestamp", time.time())
        is_win = profit > 0

        # Extrai contexto e razoes
        context = self._extract_context(trade_data)
        reasons = self._extract_reasons(trade_data, is_win)
        key = self._make_key(ativo, direction, context)

        # Inicializa combinacao se nao existe
        if key not in self.combinations:
            self.combinations[key] = {
                "wins": 0,
                "losses": 0,
                "total_profit": 0.0,
                "reasons_win": [],
                "reasons_loss": [],
                "last_trades": [],  # 1 = win, 0 = loss
                "timestamps": [],
                "created_at": timestamp,
                "last_trade_at": timestamp
            }

        combo = self.combinations[key]

        # Atualiza estatisticas
        if is_win:
            combo["wins"] += 1
            combo["reasons_win"].extend(reasons)
        else:
            combo["losses"] += 1
            combo["reasons_loss"].extend(reasons)

        combo["total_profit"] += profit
        combo["last_trades"].append(1 if is_win else 0)
        combo["timestamps"].append(timestamp)
        combo["last_trade_at"] = timestamp

        # Limita historico a 20 trades
        if len(combo["last_trades"]) > 20:
            combo["last_trades"] = combo["last_trades"][-20:]
            combo["timestamps"] = combo["timestamps"][-20:]

        # Limita razoes a 50
        combo["reasons_win"] = combo["reasons_win"][-50:]
        combo["reasons_loss"] = combo["reasons_loss"][-50:]

        # Registra no historico recente
        self.recent_trades.append({
            "key": key,
            "ativo": ativo,
            "direction": direction,
            "is_win": is_win,
            "profit": profit,
            "reasons": reasons,
            "timestamp": timestamp
        })
        self.recent_trades = self.recent_trades[-50:]

        # Verifica se deve bloquear
        self._check_block(key)

        # Salva periodicamente
        self._save()

        logger.info(f"[MEMORY] {ativo} {direction} {'WIN' if is_win else 'LOSS'} | Key={key} | Razoes={reasons}")

    def _calc_weighted_winrate(self, combo: Dict) -> float:
        """Calcula winrate com peso temporal"""
        trades = combo.get("last_trades", [])
        timestamps = combo.get("timestamps", [])

        if not trades:
            return 0.5

        now = time.time()
        weights = []

        for i, ts in enumerate(timestamps):
            hours_ago = (now - ts) / 3600

            # Peso base decai com o tempo
            days_ago = hours_ago / 24
            weight = TEMPORAL_DECAY_RATE ** days_ago

            # Bonus para trades recentes (ultimas 24h)
            if hours_ago < RECENT_WINDOW_HOURS:
                weight *= 2.0

            weights.append(max(0.1, weight))

        # Wins pesam mais
        weighted_sum = 0.0
        total_weight = 0.0

        for i, (trade, weight) in enumerate(zip(trades, weights)):
            if trade == 1:  # Win
                weighted_sum += weight * WIN_REINFORCE_WEIGHT
                total_weight += weight * WIN_REINFORCE_WEIGHT
            else:  # Loss
                weighted_sum += 0
                total_weight += weight

        return weighted_sum / max(total_weight, 1e-9)

    def _check_block(self, key: str):
        """Verifica se deve bloquear a combinacao"""
        if key not in self.combinations:
            return

        combo = self.combinations[key]
        total = combo["wins"] + combo["losses"]

        if total < MIN_TRADES_TO_BLOCK:
            return

        # Calcula winrate ponderado
        wr = self._calc_weighted_winrate(combo)

        # Verifica sequencia perdedora recente
        recent = combo["last_trades"][-LOSING_STREAK_BLOCK:]
        losing_streak = len(recent) >= LOSING_STREAK_BLOCK and sum(recent) == 0

        # Bloqueia se: WR < 35% OU sequencia perdedora
        if wr < BLOCK_WR_THRESHOLD or losing_streak:
            if key not in self.blocked:
                self.blocked.add(key)
                logger.warning(f"[MEMORY] BLOQUEADO: {key} | WR={wr*100:.0f}% | Losing Streak={losing_streak}")

        # Desbloqueia se melhorar (WR > 50% nos ultimos 5)
        elif key in self.blocked:
            recent_5 = combo["last_trades"][-5:]
            if len(recent_5) >= 3 and sum(recent_5) / len(recent_5) > 0.5:
                self.blocked.remove(key)
                logger.info(f"[MEMORY] DESBLOQUEADO: {key} | Melhoria recente")

    def should_block(self, ativo: str, direction: str, context: str = None, market_context: Dict = None) -> Tuple[bool, str]:
        """
        Verifica se deve bloquear a operacao baseado na memoria

        Retorna: (should_block, reason)
        """
        # Extrai contexto se fornecido market_context
        if context is None and market_context:
            context = self._extract_context({"market_context": market_context, "direction": direction})
        elif context is None:
            context = "default"

        key = self._make_key(ativo, direction, context)

        # Verifica bloqueio direto
        if key in self.blocked:
            combo = self.combinations.get(key, {})
            wr = self._calc_weighted_winrate(combo) if combo else 0
            return True, f"BLOCKED:{key}|WR={wr*100:.0f}%"

        # Verifica se tem historico suficiente
        if key in self.combinations:
            combo = self.combinations[key]
            total = combo["wins"] + combo["losses"]

            if total >= MIN_TRADES_TO_BLOCK:
                wr = self._calc_weighted_winrate(combo)

                # Alerta se WR baixo mas nao bloqueado
                if wr < 0.45:
                    return False, f"WARNING:LOW_WR={wr*100:.0f}%"

                # Verifica sequencia recente
                recent = combo["last_trades"][-5:]
                if len(recent) >= 3 and sum(recent) == 0:
                    return True, f"LOSING_STREAK:{len(recent)}_losses"

        return False, "APPROVED"

    def get_stats(self, ativo: str = None, direction: str = None) -> Dict:
        """Retorna estatisticas da memoria"""
        stats = {
            "total_combinations": len(self.combinations),
            "total_blocked": len(self.blocked),
            "recent_trades": len(self.recent_trades),
            "blocked_list": list(self.blocked)
        }

        if ativo:
            # Filtra por ativo
            filtered = {k: v for k, v in self.combinations.items() if ativo.upper() in k}
            stats["ativo_combinations"] = filtered

        return stats

    def get_blocked_reasons(self, key: str) -> List[str]:
        """Retorna as razoes mais comuns de loss para uma combinacao"""
        if key not in self.combinations:
            return []

        combo = self.combinations[key]
        reasons = combo.get("reasons_loss", [])

        # Conta frequencia de cada razao
        freq = {}
        for r in reasons:
            freq[r] = freq.get(r, 0) + 1

        # Ordena por frequencia
        sorted_reasons = sorted(freq.items(), key=lambda x: -x[1])
        return [f"{r}({c}x)" for r, c in sorted_reasons[:5]]

    def force_block(self, ativo: str, direction: str, context: str = "default"):
        """Forca o bloqueio de uma combinacao"""
        key = self._make_key(ativo, direction, context)
        self.blocked.add(key)
        self._save()
        logger.warning(f"[MEMORY] BLOQUEIO FORCADO: {key}")

    def force_unblock(self, ativo: str, direction: str, context: str = "default"):
        """Remove bloqueio de uma combinacao"""
        key = self._make_key(ativo, direction, context)
        if key in self.blocked:
            self.blocked.remove(key)
            self._save()
            logger.info(f"[MEMORY] DESBLOQUEIO FORCADO: {key}")

    def clear_old_data(self, days: int = 30):
        """Remove dados mais antigos que X dias"""
        cutoff = time.time() - (days * 24 * 3600)
        removed = 0

        for key in list(self.combinations.keys()):
            combo = self.combinations[key]
            if combo.get("last_trade_at", 0) < cutoff:
                del self.combinations[key]
                if key in self.blocked:
                    self.blocked.remove(key)
                removed += 1

        if removed > 0:
            self._save()
            logger.info(f"[MEMORY] Removidos {removed} combinacoes antigas (>{days} dias)")


# ===================== SINGLETON =====================
_smart_memory_instance: Optional[SmartMemory] = None


def get_smart_memory() -> SmartMemory:
    """Retorna instancia singleton da memoria"""
    global _smart_memory_instance
    if _smart_memory_instance is None:
        _smart_memory_instance = SmartMemory()
    return _smart_memory_instance


# ===================== FUNCOES AUXILIARES =====================
def should_block_trade(ativo: str, direction: str, market_context: Dict = None) -> Tuple[bool, str]:
    """Funcao helper para verificar bloqueio"""
    memory = get_smart_memory()
    return memory.should_block(ativo, direction, market_context=market_context)


def record_trade_result(ativo: str, direction: str, profit: float,
                        market_context: Dict = None, entry_quality: Dict = None,
                        padrao: str = None):
    """Funcao helper para registrar resultado"""
    memory = get_smart_memory()
    trade_data = {
        "ativo": ativo,
        "direction": direction,
        "profit": profit,
        "market_context": market_context or {},
        "entry_quality": entry_quality or {},
        "padrao": padrao,
        "timestamp": time.time()
    }
    memory.record_trade(trade_data)


if __name__ == "__main__":
    # Teste basico
    logging.basicConfig(level=logging.INFO)

    memory = SmartMemory("test_memory.json")

    # Simula alguns trades
    for i in range(10):
        memory.record_trade({
            "ativo": "EURUSD",
            "direction": "CALL",
            "profit": -5 if i < 7 else 10,  # 7 losses, 3 wins
            "market_context": {"trend": "bearish", "is_consolidating": True},
            "padrao": "ENGOLFO"
        })

    # Verifica bloqueio
    blocked, reason = memory.should_block("EURUSD", "CALL", market_context={"trend": "bearish", "is_consolidating": True})
    print(f"Bloqueado: {blocked} | Razao: {reason}")

    # Stats
    print(f"Stats: {memory.get_stats()}")
