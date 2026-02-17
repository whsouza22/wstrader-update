"""
Auto-Tuner - Sistema de Auto-Ajuste de Parametros

A IA aprende com cada resultado e equilibra os parametros automaticamente:
- Se ganha muito → pode ser mais agressiva
- Se perde muito → fica mais conservadora
- Ajusta: tolerancia S/R, min_toques, threshold CNN, etc.

Tudo persiste em arquivo para nao perder o aprendizado.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# ============================================================================
#                         CONFIGURACOES INICIAIS
# ============================================================================

# Parametros iniciais (serao ajustados pela IA)
DEFAULT_PARAMS = {
    # S/R Detection
    "tolerancia_atr": 0.25,       # 0.15 - 0.40 (quanto menor, mais preciso mas menos entradas)
    "min_toques": 2,              # 1 - 4 (quanto maior, S/R mais forte mas menos entradas)

    # Candle filters
    "max_range_atr": 1.50,        # 1.2 - 2.0 (vela esticada - quanto maior, mais entradas)
    "min_wick_frac": 0.35,        # 0.25 - 0.55 (rejeicao - quanto menor, mais entradas)

    # CNN/Risk
    "base_threshold": 0.55,       # 0.45 - 0.70 (quanto menor, mais entradas)
    "min_confidence": 0.08,       # 0.05 - 0.15

    # Timing
    "cooldown_after_loss": 3,     # 1 - 10 minutos
    "max_trades_per_hour": 8,     # 3 - 15
}

# Limites de ajuste (nao pode sair desses valores)
PARAM_LIMITS = {
    "tolerancia_atr": (0.15, 0.45),
    "min_toques": (1, 4),
    "max_range_atr": (1.15, 2.2),
    "min_wick_frac": (0.20, 0.60),
    "base_threshold": (0.40, 0.75),
    "min_confidence": (0.03, 0.20),
    "cooldown_after_loss": (1, 15),
    "max_trades_per_hour": (3, 20),
}

# Quanto ajustar por resultado
ADJUSTMENT_RATE = {
    "tolerancia_atr": 0.02,
    "min_toques": 0.2,  # Arredonda no final
    "max_range_atr": 0.05,
    "min_wick_frac": 0.03,
    "base_threshold": 0.02,
    "min_confidence": 0.01,
    "cooldown_after_loss": 0.5,
    "max_trades_per_hour": 0.5,
}

STATE_FILE = "auto_tuner_state.json"

# ============================================================================
#                         CLASSE AUTO-TUNER
# ============================================================================

class AutoTuner:
    """
    Sistema de auto-ajuste de parametros.

    Filosofia:
    - WIN → Pode ser mais agressivo (mais entradas)
    - LOSS → Fica mais conservador (menos entradas, mais filtros)
    - Sequencia de WINs → Acelera agressividade
    - Sequencia de LOSSes → Acelera conservadorismo
    """

    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self.params = DEFAULT_PARAMS.copy()

        # Historico para decisoes
        self.recent_results = []  # Lista de (timestamp, win:bool, profit:float)
        self.total_trades = 0
        self.total_wins = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0

        # Estatisticas por hora (para evitar horarios ruins)
        self.hourly_stats = {}  # {hour: {"wins": n, "losses": n}}

        # Carrega estado salvo
        self._load_state()

    def get_params(self) -> Dict[str, Any]:
        """Retorna parametros atuais para uso pelo sistema."""
        # Arredonda min_toques para inteiro
        params = self.params.copy()
        params["min_toques"] = max(1, round(params["min_toques"]))
        params["cooldown_after_loss"] = max(1, round(params["cooldown_after_loss"]))
        params["max_trades_per_hour"] = max(1, round(params["max_trades_per_hour"]))
        return params

    def on_trade_result(self, win: bool, profit: float = 0, hour: int = None):
        """
        Chamado apos cada resultado de trade.
        Ajusta parametros baseado no resultado.

        Args:
            win: True se ganhou, False se perdeu
            profit: Lucro/prejuizo do trade
            hour: Hora do trade (0-23)
        """
        now = datetime.now()
        if hour is None:
            hour = now.hour

        # Atualiza estatisticas
        self.total_trades += 1
        if win:
            self.total_wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        # Atualiza historico por hora
        h_key = str(hour)
        if h_key not in self.hourly_stats:
            self.hourly_stats[h_key] = {"wins": 0, "losses": 0}
        if win:
            self.hourly_stats[h_key]["wins"] += 1
        else:
            self.hourly_stats[h_key]["losses"] += 1

        # Guarda resultado recente (ultimas 50 operacoes)
        self.recent_results.append((now.isoformat(), win, profit))
        if len(self.recent_results) > 50:
            self.recent_results = self.recent_results[-50:]

        # === AJUSTA PARAMETROS ===
        self._adjust_params(win)

        # Salva estado
        self._save_state()

        return self.get_params()

    def _adjust_params(self, win: bool):
        """Ajusta parametros baseado no resultado."""

        # Multiplicador baseado em sequencia
        # Sequencia aumenta impacto do ajuste
        if win:
            seq_mult = min(2.0, 1.0 + (self.consecutive_wins * 0.2))
        else:
            seq_mult = min(2.0, 1.0 + (self.consecutive_losses * 0.2))

        if win:
            # === GANHOU → Pode ser mais agressivo ===
            # Mais entradas = valores que facilitam entrada

            # Tolerancia maior = mais "tocou"
            self._adjust_param("tolerancia_atr", +1, seq_mult)

            # Menos toques exigidos = mais niveis validos
            self._adjust_param("min_toques", -1, seq_mult)

            # Range maior aceito = menos filtro de vela esticada
            self._adjust_param("max_range_atr", +1, seq_mult)

            # Menos pavio exigido = mais setups
            self._adjust_param("min_wick_frac", -1, seq_mult)

            # Threshold menor = aceita probabilidade menor
            self._adjust_param("base_threshold", -1, seq_mult)

            # Menos cooldown
            self._adjust_param("cooldown_after_loss", -1, seq_mult)

            # Mais trades por hora
            self._adjust_param("max_trades_per_hour", +1, seq_mult)

        else:
            # === PERDEU → Fica mais conservador ===
            # Menos entradas = valores que dificultam entrada

            # Tolerancia menor = toque mais preciso
            self._adjust_param("tolerancia_atr", -1, seq_mult)

            # Mais toques exigidos = S/R mais forte
            self._adjust_param("min_toques", +1, seq_mult)

            # Range menor aceito = filtra velas esticadas
            self._adjust_param("max_range_atr", -1, seq_mult)

            # Mais pavio exigido = melhor rejeicao
            self._adjust_param("min_wick_frac", +1, seq_mult)

            # Threshold maior = exige mais certeza
            self._adjust_param("base_threshold", +1, seq_mult)

            # Mais cooldown apos loss
            self._adjust_param("cooldown_after_loss", +1, seq_mult)

            # Menos trades por hora
            self._adjust_param("max_trades_per_hour", -1, seq_mult)

    def _adjust_param(self, param: str, direction: int, multiplier: float):
        """
        Ajusta um parametro especifico.

        Args:
            param: Nome do parametro
            direction: +1 para aumentar, -1 para diminuir
            multiplier: Multiplicador do ajuste
        """
        if param not in self.params:
            return

        rate = ADJUSTMENT_RATE.get(param, 0.01)
        min_val, max_val = PARAM_LIMITS.get(param, (0, 1))

        # Calcula ajuste
        adjustment = direction * rate * multiplier
        new_val = self.params[param] + adjustment

        # Aplica limites
        new_val = max(min_val, min(max_val, new_val))
        self.params[param] = new_val

    def is_good_hour(self, hour: int = None) -> bool:
        """Verifica se a hora atual tem bom historico."""
        if hour is None:
            hour = datetime.now().hour

        h_key = str(hour)
        if h_key not in self.hourly_stats:
            return True  # Sem dados = assume bom

        stats = self.hourly_stats[h_key]
        total = stats["wins"] + stats["losses"]

        if total < 5:
            return True  # Poucos dados = assume bom

        win_rate = stats["wins"] / total
        return win_rate >= 0.45  # Aceita ate 45% de winrate

    def get_hour_adjustment(self, hour: int = None) -> float:
        """
        Retorna ajuste de threshold baseado na hora.
        Horas ruins = threshold mais alto.
        """
        if hour is None:
            hour = datetime.now().hour

        h_key = str(hour)
        if h_key not in self.hourly_stats:
            return 0.0

        stats = self.hourly_stats[h_key]
        total = stats["wins"] + stats["losses"]

        if total < 5:
            return 0.0

        win_rate = stats["wins"] / total

        # Hora muito ruim = aumenta threshold em ate 0.10
        if win_rate < 0.35:
            return 0.10
        elif win_rate < 0.45:
            return 0.05
        elif win_rate > 0.60:
            return -0.03  # Hora boa = diminui um pouco

        return 0.0

    def get_stats(self) -> Dict:
        """Retorna estatisticas completas."""
        # Calcula win rate recente (ultimas 20 ops)
        recent_20 = self.recent_results[-20:] if len(self.recent_results) >= 20 else self.recent_results
        recent_wins = sum(1 for _, win, _ in recent_20 if win)
        recent_wr = recent_wins / len(recent_20) if recent_20 else 0

        return {
            "params": self.get_params(),
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_trades - self.total_wins,
            "win_rate": self.total_wins / self.total_trades if self.total_trades > 0 else 0,
            "recent_win_rate": recent_wr,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "hourly_stats": self.hourly_stats,
            "current_hour": datetime.now().hour,
            "is_good_hour": self.is_good_hour(),
        }

    def reset(self):
        """Reseta para valores padrao."""
        self.params = DEFAULT_PARAMS.copy()
        self.recent_results = []
        self.total_trades = 0
        self.total_wins = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.hourly_stats = {}
        self._save_state()
        print("[AUTO-TUNER] Reset para valores padrao")

    def _load_state(self):
        """Carrega estado salvo."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)

                # Carrega parametros (mescla com default para novos params)
                saved_params = state.get("params", {})
                for k, v in saved_params.items():
                    if k in self.params:
                        self.params[k] = v

                self.recent_results = state.get("recent_results", [])
                self.total_trades = state.get("total_trades", 0)
                self.total_wins = state.get("total_wins", 0)
                self.consecutive_wins = state.get("consecutive_wins", 0)
                self.consecutive_losses = state.get("consecutive_losses", 0)
                self.hourly_stats = state.get("hourly_stats", {})

                print(f"[AUTO-TUNER] Estado carregado: {self.total_trades} trades, WR={self.total_wins/max(1,self.total_trades)*100:.1f}%")

            except Exception as e:
                print(f"[AUTO-TUNER] Erro ao carregar estado: {e}")

    def _save_state(self):
        """Salva estado no disco."""
        try:
            state = {
                "params": self.params,
                "recent_results": self.recent_results[-50:],
                "total_trades": self.total_trades,
                "total_wins": self.total_wins,
                "consecutive_wins": self.consecutive_wins,
                "consecutive_losses": self.consecutive_losses,
                "hourly_stats": self.hourly_stats,
                "last_updated": datetime.now().isoformat(),
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            print(f"[AUTO-TUNER] Erro ao salvar estado: {e}")


# ============================================================================
#                         INSTANCIA GLOBAL
# ============================================================================

_tuner_instance: Optional[AutoTuner] = None

def get_tuner() -> AutoTuner:
    """Retorna instancia global do AutoTuner."""
    global _tuner_instance
    if _tuner_instance is None:
        _tuner_instance = AutoTuner()
    return _tuner_instance


# ============================================================================
#                         TESTE LOCAL
# ============================================================================

if __name__ == "__main__":
    tuner = get_tuner()

    print("=== Estado Inicial ===")
    stats = tuner.get_stats()
    print(f"Params: {stats['params']}")
    print(f"Total trades: {stats['total_trades']}, WR: {stats['win_rate']*100:.1f}%")

    print("\n=== Simulando 3 WINS ===")
    for i in range(3):
        tuner.on_trade_result(win=True, profit=10)
        print(f"Win {i+1}: threshold={tuner.params['base_threshold']:.3f}, tol={tuner.params['tolerancia_atr']:.3f}")

    print("\n=== Simulando 2 LOSSES ===")
    for i in range(2):
        tuner.on_trade_result(win=False, profit=-10)
        print(f"Loss {i+1}: threshold={tuner.params['base_threshold']:.3f}, tol={tuner.params['tolerancia_atr']:.3f}")

    print("\n=== Estado Final ===")
    stats = tuner.get_stats()
    print(f"Params: {stats['params']}")
    print(f"Total trades: {stats['total_trades']}, WR: {stats['win_rate']*100:.1f}%")
