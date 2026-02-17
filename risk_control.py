"""
Risk Control - Gate + Controles Pos-Modelo

Controles aplicados APOS o modelo CNN decidir:
1. Threshold dinamico (sobe apos loss, desce apos win)
2. Limite de trades por janela
3. Cooldown apos loss (evita tilt)
4. Pausa automatica se performance cair

Threshold inicial: 0.60 (escolhido pelo usuario)
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List

# ============================================================================
#                         CONFIGURACOES
# ============================================================================

# Threshold
BASE_THRESHOLD = 0.60           # Threshold inicial
MAX_THRESHOLD = 0.80            # Maximo em mercado ruim
MIN_THRESHOLD = 0.55            # Minimo em sequencia de wins
THRESHOLD_UP_ON_LOSS = 0.05     # Sobe 5% no loss
THRESHOLD_DOWN_ON_WIN = 0.02    # Desce 2% no win

# Limite de trades
MAX_TRADES_PER_WINDOW = 3       # Max trades por janela
WINDOW_MINUTES = 30             # Janela de 30 min

# Cooldown
COOLDOWN_MINUTES = 5            # Pausa de 5 min apos loss
COOLDOWN_CONSECUTIVE = 2        # Ativa cooldown estendido apos 2 losses seguidos
EXTENDED_COOLDOWN_MINUTES = 10  # Cooldown estendido

# Pausa automatica
MAX_CONSECUTIVE_LOSSES = 3      # Pausa apos 3 losses seguidos
DISABLE_AUTO_PAUSE = os.getenv("WS_RISK_DISABLE_PAUSE", "0").strip() == "1"
PAUSE_MINUTES = 0 if DISABLE_AUTO_PAUSE else 30  # Pausa de 30 min (ou desativada)

# Persistencia
RISK_STATE_FILE = "risk_control_state.json"

# ============================================================================
#                         CLASSE RISK CONTROL
# ============================================================================

class RiskControl:
    """
    Controle de risco pos-modelo.

    Aplica filtros apos o modelo CNN decidir, incluindo:
    - Threshold dinamico
    - Limite de trades por janela
    - Cooldown apos loss
    - Pausa automatica em sequencia de losses
    """

    def __init__(
        self,
        base_threshold: float = BASE_THRESHOLD,
        max_threshold: float = MAX_THRESHOLD,
        min_threshold: float = MIN_THRESHOLD,
        threshold_up_on_loss: float = THRESHOLD_UP_ON_LOSS,
        threshold_down_on_win: float = THRESHOLD_DOWN_ON_WIN,
        max_trades_per_window: int = MAX_TRADES_PER_WINDOW,
        window_minutes: int = WINDOW_MINUTES,
        cooldown_minutes: int = COOLDOWN_MINUTES,
        extended_cooldown_minutes: int = EXTENDED_COOLDOWN_MINUTES,
        max_consecutive_losses: int = MAX_CONSECUTIVE_LOSSES,
        pause_minutes: int = PAUSE_MINUTES,
        state_file: str = RISK_STATE_FILE
    ):
        # Parametros
        self.base_threshold = base_threshold
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.threshold_up_on_loss = threshold_up_on_loss
        self.threshold_down_on_win = threshold_down_on_win
        self.max_trades_per_window = max_trades_per_window
        self.window_minutes = window_minutes
        self.cooldown_minutes = cooldown_minutes
        self.extended_cooldown_minutes = extended_cooldown_minutes
        self.max_consecutive_losses = max_consecutive_losses
        self.pause_minutes = pause_minutes
        self.state_file = state_file

        # Estado
        self.current_threshold = base_threshold
        self.trades_in_window: List[datetime] = []
        self.recent_results: List[bool] = []   # True = win, False = loss
        self.consecutive_losses = 0
        self.cooldown_until: Optional[datetime] = None
        self.pause_until: Optional[datetime] = None

        # Estatisticas
        self.total_trades = 0
        self.total_wins = 0
        self.total_blocked = 0
        self.block_reasons: Dict[str, int] = {}

        # Carrega estado salvo
        self._load_state()

    def should_trade(self, prediction: Dict) -> Tuple[bool, str, Dict]:
        """
        Verifica se deve operar com base no resultado do modelo CNN.

        Args:
            prediction: Dict do modelo CNN com:
                - class: "CALL", "PUT", "NO_TRADE"
                - probability: 0.0-1.0
                - raw_probs: [p_call, p_put, p_notrade]
                - confidence: diferenca entre top 2

        Returns:
            Tuple[bool, str, Dict]: (pode_operar, motivo, detalhes)
        """
        now = datetime.now()
        details = {
            "current_threshold": self.current_threshold,
            "consecutive_losses": self.consecutive_losses,
            "trades_in_window": len(self._get_trades_in_window())
        }

        # 1. Modelo disse NO_TRADE
        if prediction.get("class") == "NO_TRADE":
            self._record_block("MODEL_NO_TRADE")
            return False, "MODEL_NO_TRADE", details

        # 2. Em pausa automatica
        if self.pause_until and now < self.pause_until:
            if self.pause_minutes <= 0:
                # Pausa desativada via env; ignora e limpa
                self.pause_until = None
            else:
                remaining = (self.pause_until - now).seconds // 60
                details["pause_remaining_min"] = remaining
                self._record_block("AUTO_PAUSE")
                return False, f"AUTO_PAUSE ({remaining}min restantes)", details

        # 3. Em cooldown
        if self.cooldown_until and now < self.cooldown_until:
            remaining = (self.cooldown_until - now).seconds
            details["cooldown_remaining_sec"] = remaining
            self._record_block("COOLDOWN_ACTIVE")
            return False, f"COOLDOWN_ACTIVE ({remaining}s restantes)", details

        # 4. Probabilidade abaixo do threshold
        prob = prediction.get("probability", 0)
        details["probability"] = prob
        if prob < self.current_threshold:
            self._record_block("LOW_PROBABILITY")
            return False, f"LOW_PROBABILITY ({prob:.2f} < {self.current_threshold:.2f})", details

        # 5. Limite de trades na janela
        trades_in_window = self._get_trades_in_window()
        if len(trades_in_window) >= self.max_trades_per_window:
            self._record_block("MAX_TRADES_WINDOW")
            return False, f"MAX_TRADES_WINDOW ({len(trades_in_window)}/{self.max_trades_per_window})", details

        # 6. Confianca muito baixa
        confidence = prediction.get("confidence", 0)
        details["confidence"] = confidence
        if confidence < 0.10:  # Menos de 10% de diferenca entre top 2
            self._record_block("LOW_CONFIDENCE")
            return False, f"LOW_CONFIDENCE ({confidence:.2f})", details

        return True, "OK", details

    def on_trade_opened(self):
        """Registra que um trade foi aberto."""
        self.trades_in_window.append(datetime.now())
        self.total_trades += 1
        self._save_state()

    def on_result(self, win: bool):
        """
        Atualiza controles apos resultado do trade.

        Args:
            win: True se ganhou, False se perdeu
        """
        now = datetime.now()
        self.recent_results.append(win)

        # Mantem apenas ultimos 20 resultados
        if len(self.recent_results) > 20:
            self.recent_results = self.recent_results[-20:]

        if win:
            # === WIN ===
            self.total_wins += 1
            self.consecutive_losses = 0

            # Abaixa threshold (recompensa)
            self.current_threshold = max(
                self.min_threshold,
                self.current_threshold - self.threshold_down_on_win
            )

            # Limpa cooldown
            self.cooldown_until = None

        else:
            # === LOSS ===
            self.consecutive_losses += 1

            # Sobe threshold (penalidade)
            self.current_threshold = min(
                self.max_threshold,
                self.current_threshold + self.threshold_up_on_loss
            )

            # Ativa cooldown
            if self.consecutive_losses >= COOLDOWN_CONSECUTIVE:
                # Cooldown estendido apos losses consecutivos
                cooldown_min = self.extended_cooldown_minutes
            else:
                cooldown_min = self.cooldown_minutes

            self.cooldown_until = now + timedelta(minutes=cooldown_min)

            # Pausa automatica apos muitos losses seguidos
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.pause_until = now + timedelta(minutes=self.pause_minutes)
                print(f"[RISK] Pausa automatica ativada por {self.pause_minutes} min")

        self._save_state()

    def _get_trades_in_window(self) -> List[datetime]:
        """Retorna trades dentro da janela de tempo atual."""
        now = datetime.now()
        window_start = now - timedelta(minutes=self.window_minutes)

        # Filtra trades antigos
        self.trades_in_window = [
            t for t in self.trades_in_window if t > window_start
        ]

        return self.trades_in_window

    def _record_block(self, reason: str):
        """Registra motivo de bloqueio para estatisticas."""
        self.total_blocked += 1
        self.block_reasons[reason] = self.block_reasons.get(reason, 0) + 1

    def reset_cooldown(self):
        """Reseta cooldown manualmente (para debug)."""
        self.cooldown_until = None
        self._save_state()

    def reset_pause(self):
        """Reseta pausa automatica manualmente."""
        self.pause_until = None
        self.consecutive_losses = 0
        self._save_state()

    def reset_threshold(self):
        """Reseta threshold para valor base."""
        self.current_threshold = self.base_threshold
        self._save_state()

    def get_stats(self) -> Dict:
        """Retorna estatisticas do controle de risco."""
        now = datetime.now()

        return {
            "current_threshold": self.current_threshold,
            "base_threshold": self.base_threshold,
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_trades - self.total_wins,
            "win_rate": self.total_wins / self.total_trades if self.total_trades > 0 else 0,
            "consecutive_losses": self.consecutive_losses,
            "trades_in_window": len(self._get_trades_in_window()),
            "max_trades_per_window": self.max_trades_per_window,
            "in_cooldown": self.cooldown_until is not None and now < self.cooldown_until,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "in_pause": self.pause_until is not None and now < self.pause_until,
            "pause_until": self.pause_until.isoformat() if self.pause_until else None,
            "total_blocked": self.total_blocked,
            "block_reasons": self.block_reasons,
            "recent_results": self.recent_results[-10:]  # Ultimos 10
        }

    def _load_state(self):
        """Carrega estado salvo do disco."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)

                self.current_threshold = state.get('current_threshold', self.base_threshold)
                self.consecutive_losses = state.get('consecutive_losses', 0)
                self.total_trades = state.get('total_trades', 0)
                self.total_wins = state.get('total_wins', 0)
                self.total_blocked = state.get('total_blocked', 0)
                self.block_reasons = state.get('block_reasons', {})
                self.recent_results = state.get('recent_results', [])

                # Carrega timestamps
                if state.get('cooldown_until'):
                    self.cooldown_until = datetime.fromisoformat(state['cooldown_until'])
                if state.get('pause_until'):
                    self.pause_until = datetime.fromisoformat(state['pause_until'])

                print(f"[RISK] Estado carregado: threshold={self.current_threshold:.2f}")

            except Exception as e:
                print(f"[RISK] Erro ao carregar estado: {e}")

    def _save_state(self):
        """Salva estado no disco."""
        try:
            state = {
                'current_threshold': self.current_threshold,
                'consecutive_losses': self.consecutive_losses,
                'total_trades': self.total_trades,
                'total_wins': self.total_wins,
                'total_blocked': self.total_blocked,
                'block_reasons': self.block_reasons,
                'recent_results': self.recent_results[-20:],
                'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None,
                'pause_until': self.pause_until.isoformat() if self.pause_until else None,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            print(f"[RISK] Erro ao salvar estado: {e}")


# ============================================================================
#                         FUNCOES AUXILIARES
# ============================================================================

def create_risk_control(**kwargs) -> RiskControl:
    """Factory function para criar RiskControl."""
    return RiskControl(**kwargs)


# ============================================================================
#                         TESTE LOCAL
# ============================================================================

if __name__ == "__main__":
    # Teste basico
    rc = RiskControl()

    print("=== Estado Inicial ===")
    print(f"Threshold: {rc.current_threshold}")

    # Simula predicao do modelo
    prediction_ok = {
        "class": "CALL",
        "probability": 0.75,
        "raw_probs": [0.75, 0.15, 0.10],
        "confidence": 0.60
    }

    prediction_low = {
        "class": "PUT",
        "probability": 0.45,
        "raw_probs": [0.30, 0.45, 0.25],
        "confidence": 0.15
    }

    prediction_notrade = {
        "class": "NO_TRADE",
        "probability": 0.50,
        "raw_probs": [0.25, 0.25, 0.50],
        "confidence": 0.25
    }

    print("\n=== Teste: Predicao OK ===")
    can_trade, reason, details = rc.should_trade(prediction_ok)
    print(f"Pode operar: {can_trade}, Motivo: {reason}")

    print("\n=== Teste: Predicao Baixa Prob ===")
    can_trade, reason, details = rc.should_trade(prediction_low)
    print(f"Pode operar: {can_trade}, Motivo: {reason}")

    print("\n=== Teste: NO_TRADE ===")
    can_trade, reason, details = rc.should_trade(prediction_notrade)
    print(f"Pode operar: {can_trade}, Motivo: {reason}")

    print("\n=== Simula 3 Losses ===")
    for i in range(3):
        rc.on_trade_opened()
        rc.on_result(win=False)
        print(f"Loss {i+1}: threshold={rc.current_threshold:.2f}, consecutive={rc.consecutive_losses}")

    print("\n=== Estatisticas ===")
    stats = rc.get_stats()
    for k, v in stats.items():
        print(f"{k}: {v}")
