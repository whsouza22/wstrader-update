"""
ws_data_manager.py — Gerenciador centralizado de dados do WS Trader.

Consolida TODOS os dados de IA/trading em UM ÚNICO arquivo JSON:
  ws_trading_data.json

Substitui os arquivos espalhados:
  - ws_ai_stats_{m1,bullex,casatrader}.json     → seção "ai_stats"
  - ws_lgbm_data_{m1,bullex,casatrader}.json     → seção "lgbm_data"
  - ws_backtest_history_{m1,bullex,casatrader}.json → seção "backtest_history"
  - ws_loss_memory.json                           → seção "loss_memory"
  - auto_tuner_state.json                         → seção "auto_tuner"

Arquivos que continuam separados (binário):
  - ws_lgbm_model_{broker}.pkl  (pickle, não cabe em JSON)

Uso:
  from ws_data_manager import DataManager
  dm = DataManager()                    # carrega tudo
  stats = dm.get_ai_stats("m1")        # lê seção
  dm.set_ai_stats("m1", stats)         # grava seção (auto-save)
  dm.clean_all()                       # limpa TUDO de uma vez
"""

import os
import json
import time
import logging
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime

log = logging.getLogger("WS_AUTO_AI")

# ========================= ARQUIVO ÚNICO =========================
# Salvar na pasta do usuário (~/.wstrader/) para sobreviver a reinstalações
_USER_DATA_DIR = os.path.join(os.path.expanduser("~"), ".wstrader")
os.makedirs(_USER_DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(_USER_DATA_DIR, "ws_trading_data.json")

# Caminho legado (pasta de instalação) — usado para migração automática
_LEGACY_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ws_trading_data.json")

# Lock para acesso thread-safe (RLock permite reentrada no mesmo thread)
_lock = threading.RLock()

# Estrutura padrão do arquivo consolidado
_DEFAULT_DATA = {
    "version": 2,
    "last_updated": "",
    "brokers": {
        "m1": {
            "ai_stats": {"meta": {"total": 0}, "arms": {}},
            "lgbm_data": [],
            "backtest_history": [],
            "session": {"total": 0, "wins": 0, "last_updated": ""},
        },
        "bullex": {
            "ai_stats": {"meta": {"total": 0}, "arms": {}},
            "lgbm_data": [],
            "backtest_history": [],
            "session": {"total": 0, "wins": 0, "last_updated": ""},
        },
        "casatrader": {
            "ai_stats": {"meta": {"total": 0}, "arms": {}},
            "lgbm_data": [],
            "backtest_history": [],
            "session": {"total": 0, "wins": 0, "last_updated": ""},
        },
        # ── UNIFICADO: IA compartilhada entre todas as corretoras ──
        # Gráficos OTC são idênticos → treinamento unificado = 3x mais rápido
        "unified": {
            "ai_stats": {"meta": {"total": 0}, "arms": {}},
            "lgbm_data": [],
            "backtest_history": [],
            "session": {"total": 0, "wins": 0, "last_updated": ""},
        },
    },
    "loss_memory": [],
    "auto_tuner": {},
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge profundo: base recebe valores de override sem perder chaves extras."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _ensure_structure(data: dict) -> dict:
    """Garante que o dict tem todas as chaves obrigatórias."""
    import copy
    default = copy.deepcopy(_DEFAULT_DATA)
    return _deep_merge(default, data)


class DataManager:
    """Singleton de acesso ao arquivo consolidado ws_trading_data.json."""

    _instance: Optional["DataManager"] = None
    _data: Dict[str, Any] = {}
    _dirty: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self, auto_load: bool = True):
        if not self._loaded and auto_load:
            self.load()

    # ========================= LOAD / SAVE =========================

    def load(self):
        """Carrega o arquivo consolidado (ou migra dos arquivos antigos)."""
        with _lock:
            # ── Auto-migrar da pasta de instalação para ~/.wstrader/ ──
            if not os.path.exists(DATA_FILE) and os.path.exists(_LEGACY_DATA_FILE):
                try:
                    import shutil
                    shutil.copy2(_LEGACY_DATA_FILE, DATA_FILE)
                    log.info(f"[DATA] ✅ Migrado ws_trading_data.json da pasta de instalação para {_USER_DATA_DIR}")
                except Exception as e:
                    log.warning(f"[DATA] Erro ao migrar arquivo legado: {e}")

            # ── Migrar modelos .pkl para ~/.wstrader/ ──
            _install_base = os.path.dirname(os.path.abspath(__file__))
            for suffix in ["m1", "bullex", "casatrader", "unified"]:
                old_pkl = os.path.join(_install_base, f"ws_lgbm_model_{suffix}.pkl")
                new_pkl = os.path.join(_USER_DATA_DIR, f"ws_lgbm_model_{suffix}.pkl")
                if os.path.exists(old_pkl) and not os.path.exists(new_pkl):
                    try:
                        import shutil
                        shutil.copy2(old_pkl, new_pkl)
                        log.info(f"[DATA] ✅ Modelo migrado: ws_lgbm_model_{suffix}.pkl → {_USER_DATA_DIR}")
                    except Exception:
                        pass

            if os.path.exists(DATA_FILE):
                try:
                    with open(DATA_FILE, "r", encoding="utf-8") as f:
                        raw = f.read().strip()
                    if raw and len(raw) > 5:
                        self._data = json.loads(raw)
                        self._data = _ensure_structure(self._data)
                        # ── Auto-migrar dados separados → unified ──
                        self._migrate_to_unified()
                        self._loaded = True
                        log.info(f"[DATA] Arquivo consolidado carregado: {DATA_FILE}")
                        return
                except (json.JSONDecodeError, Exception) as e:
                    log.warning(f"[DATA] Erro ao carregar {DATA_FILE}: {e} — tentando migração")

            # Arquivo não existe ou corrompido → migrar dos antigos
            self._migrate_from_legacy()
            self._loaded = True

    def save(self, force: bool = False):
        """Salva o arquivo consolidado (escrita atômica)."""
        with _lock:
            self._data["last_updated"] = datetime.now().isoformat()
            tmp = DATA_FILE + ".tmp"
            try:
                raw = json.dumps(self._data, ensure_ascii=False, indent=2)
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(raw)
                    f.flush()
                    os.fsync(f.fileno())
                if os.path.exists(DATA_FILE):
                    os.remove(DATA_FILE)
                os.rename(tmp, DATA_FILE)
                self._dirty = False
            except Exception as e:
                log.warning(f"[DATA] Erro ao salvar: {e}")
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass

    # ========================= MIGRAÇÃO DOS ARQUIVOS ANTIGOS =========================

    def _migrate_from_legacy(self):
        """Importa dados dos arquivos JSON antigos para o arquivo consolidado."""
        import copy
        self._data = copy.deepcopy(_DEFAULT_DATA)
        base = os.path.dirname(os.path.abspath(__file__))
        migrated_any = False

        for suffix in ["m1", "bullex", "casatrader"]:
            # AI Stats (Bayesian)
            path = os.path.join(base, f"ws_ai_stats_{suffix}.json")
            data = self._read_legacy_json(path)
            if data:
                self._data["brokers"][suffix]["ai_stats"] = data
                migrated_any = True

            # LGBM Data
            path = os.path.join(base, f"ws_lgbm_data_{suffix}.json")
            data = self._read_legacy_json(path)
            if isinstance(data, list):
                self._data["brokers"][suffix]["lgbm_data"] = data
                migrated_any = True

            # Backtest History
            path = os.path.join(base, f"ws_backtest_history_{suffix}.json")
            data = self._read_legacy_json(path)
            if isinstance(data, list):
                self._data["brokers"][suffix]["backtest_history"] = data
                migrated_any = True

        # Loss Memory
        path = os.path.join(base, "ws_loss_memory.json")
        data = self._read_legacy_json(path)
        if isinstance(data, list):
            self._data["loss_memory"] = data
            migrated_any = True

        # Auto Tuner
        path = os.path.join(base, "auto_tuner_state.json")
        data = self._read_legacy_json(path)
        if isinstance(data, dict):
            self._data["auto_tuner"] = data
            migrated_any = True

        if migrated_any:
            log.info("[DATA] ✅ Migração de arquivos antigos concluída → ws_trading_data.json")
            self.save()
            # Renomear arquivos antigos para .bak (não deleta)
            self._backup_legacy_files()
        else:
            log.info("[DATA] Nenhum arquivo antigo encontrado — criando arquivo vazio")
            self.save()

    def _migrate_to_unified(self):
        """
        Migra dados de ai_stats e lgbm_data dos brokers separados (m1, bullex, casatrader)
        para a seção 'unified'. Faz merge dos arms Bayesianos somando a/b/n.
        Só executa se unified estiver vazio e algum broker tiver dados.
        """
        brokers = self._data.get("brokers", {})
        unified = brokers.get("unified", {})
        unified_arms = unified.get("ai_stats", {}).get("arms", {})
        unified_lgbm = unified.get("lgbm_data", [])

        # Se unified já tem dados, não migrar de novo
        if unified_arms or unified_lgbm:
            return

        # Verificar se algum broker tem dados para migrar
        has_data = False
        for suffix in ["m1", "bullex", "casatrader"]:
            bk = brokers.get(suffix, {})
            if bk.get("ai_stats", {}).get("arms", {}):
                has_data = True
                break
            if bk.get("lgbm_data", []):
                has_data = True
                break

        if not has_data:
            return

        # ── Merge Bayesian arms: soma a, b, n de todos os brokers ──
        merged_arms = {}
        merged_patterns = {}
        total_trades = 0

        for suffix in ["m1", "bullex", "casatrader"]:
            bk = brokers.get(suffix, {})
            ai = bk.get("ai_stats", {})
            arms = ai.get("arms", {})
            patterns = ai.get("patterns", {})
            total_trades += ai.get("meta", {}).get("total", 0)

            for key, arm in arms.items():
                if key not in merged_arms:
                    merged_arms[key] = {"a": 0.0, "b": 0.0, "n": 0}
                merged_arms[key]["a"] += float(arm.get("a", 0.0))
                merged_arms[key]["b"] += float(arm.get("b", 0.0))
                merged_arms[key]["n"] += int(arm.get("n", 0))

            for key, pat in patterns.items():
                if key not in merged_patterns:
                    merged_patterns[key] = {"trades": 0, "wins": 0, "losses": 0}
                merged_patterns[key]["trades"] += int(pat.get("trades", 0))
                merged_patterns[key]["wins"] += int(pat.get("wins", 0))
                merged_patterns[key]["losses"] += int(pat.get("losses", 0))

        # ── Merge LGBM data: concatena amostras de todos os brokers ──
        merged_lgbm = []
        for suffix in ["m1", "bullex", "casatrader"]:
            bk = brokers.get(suffix, {})
            lgbm = bk.get("lgbm_data", [])
            if isinstance(lgbm, list):
                merged_lgbm.extend(lgbm)

        # ── Gravar no unified ──
        self._ensure_broker("unified")
        self._data["brokers"]["unified"]["ai_stats"] = {
            "meta": {"total": total_trades},
            "arms": merged_arms,
            "patterns": merged_patterns,
        }
        self._data["brokers"]["unified"]["lgbm_data"] = merged_lgbm

        log.info(
            f"[DATA] ✅ Migração UNIFIED concluída: "
            f"{len(merged_arms)} arms | {total_trades} trades | "
            f"{len(merged_lgbm)} LGBM samples (de m1+bullex+casatrader)"
        )
        self.save()

    def _read_legacy_json(self, path: str):
        """Lê um arquivo JSON legado. Retorna None se falhar."""
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            log.warning(f"[DATA] Erro ao ler legado {path}: {e}")
        return None

    def _backup_legacy_files(self):
        """Renomeia arquivos antigos para .bak após migração."""
        base = os.path.dirname(os.path.abspath(__file__))
        legacy_files = []
        for suffix in ["m1", "bullex", "casatrader"]:
            legacy_files.append(f"ws_ai_stats_{suffix}.json")
            legacy_files.append(f"ws_lgbm_data_{suffix}.json")
            legacy_files.append(f"ws_backtest_history_{suffix}.json")
        legacy_files.append("ws_loss_memory.json")
        legacy_files.append("auto_tuner_state.json")

        for fname in legacy_files:
            fpath = os.path.join(base, fname)
            if os.path.exists(fpath):
                bak = fpath + ".migrated.bak"
                try:
                    os.rename(fpath, bak)
                    log.info(f"[DATA] Arquivo legado renomeado: {fname} → {fname}.migrated.bak")
                except Exception as e:
                    log.warning(f"[DATA] Não conseguiu renomear {fname}: {e}")

    # ========================= AI STATS (Bayesian) =========================

    def get_ai_stats(self, broker_suffix: str) -> Dict[str, Any]:
        """Retorna stats Bayesian para o broker."""
        return self._data.get("brokers", {}).get(broker_suffix, {}).get(
            "ai_stats", {"meta": {"total": 0}, "arms": {}}
        )

    def set_ai_stats(self, broker_suffix: str, stats: Dict[str, Any]):
        """Salva stats Bayesian e persiste."""
        self._ensure_broker(broker_suffix)
        self._data["brokers"][broker_suffix]["ai_stats"] = stats
        self.save()

    # ========================= LGBM DATA =========================

    def get_lgbm_data(self, broker_suffix: str) -> List[Dict]:
        """Retorna lista de amostras LGBM do broker."""
        return self._data.get("brokers", {}).get(broker_suffix, {}).get("lgbm_data", [])

    def set_lgbm_data(self, broker_suffix: str, data: List[Dict]):
        """Salva amostras LGBM e persiste."""
        self._ensure_broker(broker_suffix)
        self._data["brokers"][broker_suffix]["lgbm_data"] = data
        self.save()

    # ========================= BACKTEST HISTORY =========================

    def get_backtest_history(self, broker_suffix: str) -> List[Dict]:
        """Retorna histórico de backtests do broker."""
        return self._data.get("brokers", {}).get(broker_suffix, {}).get("backtest_history", [])

    def set_backtest_history(self, broker_suffix: str, history: List[Dict]):
        """Salva histórico de backtests e persiste."""
        self._ensure_broker(broker_suffix)
        self._data["brokers"][broker_suffix]["backtest_history"] = history
        self.save()

    # ========================= SESSION STATE (NOVO — FIX RESTART) =========================

    def get_session(self, broker_suffix: str) -> Dict[str, Any]:
        """Retorna estado da sessão (total, wins) — sobrevive a restarts."""
        return self._data.get("brokers", {}).get(broker_suffix, {}).get(
            "session", {"total": 0, "wins": 0, "last_updated": ""}
        )

    def set_session(self, broker_suffix: str, total: int, wins: int):
        """Salva estado da sessão e persiste."""
        self._ensure_broker(broker_suffix)
        self._data["brokers"][broker_suffix]["session"] = {
            "total": total,
            "wins": wins,
            "last_updated": datetime.now().isoformat(),
        }
        self.save()

    # ========================= LOSS MEMORY =========================

    def get_loss_memory(self) -> List[Dict]:
        """Retorna lista de registros de loss memory."""
        return self._data.get("loss_memory", [])

    def set_loss_memory(self, records: List[Dict]):
        """Salva loss memory e persiste."""
        self._data["loss_memory"] = records
        self.save()

    # ========================= AUTO TUNER =========================

    def get_auto_tuner(self) -> Dict[str, Any]:
        """Retorna estado do auto tuner."""
        return self._data.get("auto_tuner", {})

    def set_auto_tuner(self, state: Dict[str, Any]):
        """Salva estado do auto tuner e persiste."""
        self._data["auto_tuner"] = state
        self.save()

    # ========================= LIMPEZA GLOBAL =========================

    def clean_all(self):
        """Limpa TODOS os dados de uma vez — um único ponto de limpeza."""
        import copy
        self._data = copy.deepcopy(_DEFAULT_DATA)
        self.save()
        # Também limpar modelos LGBM (pkl) — em ~/.wstrader/ e na pasta de instalação (legado)
        for search_dir in [_USER_DATA_DIR, os.path.dirname(os.path.abspath(__file__))]:
            for suffix in ["m1", "bullex", "casatrader", "unified"]:
                pkl = os.path.join(search_dir, f"ws_lgbm_model_{suffix}.pkl")
                if os.path.exists(pkl):
                    try:
                        os.remove(pkl)
                        log.info(f"[DATA] Modelo removido: ws_lgbm_model_{suffix}.pkl")
                    except Exception:
                        pass
        log.info("[DATA] ✅ TODOS os dados limpos com sucesso (ws_trading_data.json zerado)")

    def clean_broker(self, broker_suffix: str):
        """Limpa dados de um broker específico."""
        import copy
        default_broker = copy.deepcopy(_DEFAULT_DATA["brokers"]["m1"])
        self._data["brokers"][broker_suffix] = default_broker
        self.save()
        # Limpar modelo pkl desse broker — em ~/.wstrader/ e na pasta de instalação (legado)
        for search_dir in [_USER_DATA_DIR, os.path.dirname(os.path.abspath(__file__))]:
            pkl = os.path.join(search_dir, f"ws_lgbm_model_{broker_suffix}.pkl")
            if os.path.exists(pkl):
                try:
                    os.remove(pkl)
                except Exception:
                    pass
        log.info(f"[DATA] ✅ Dados do broker '{broker_suffix}' limpos")

    # ========================= HELPERS =========================

    def _ensure_broker(self, suffix: str):
        """Garante que a seção do broker existe."""
        if "brokers" not in self._data:
            self._data["brokers"] = {}
        if suffix not in self._data["brokers"]:
            import copy
            self._data["brokers"][suffix] = copy.deepcopy(_DEFAULT_DATA["brokers"]["m1"])

    def get_summary(self) -> str:
        """Retorna resumo textual de todos os dados (para debug/chat)."""
        lines = ["📊 Dados consolidados (ws_trading_data.json):"]
        # Mostrar dados UNIFIED primeiro (principal)
        unified = self._data.get("brokers", {}).get("unified", {})
        u_ai = unified.get("ai_stats", {}).get("meta", {})
        u_lgbm = unified.get("lgbm_data", [])
        u_arms = unified.get("ai_stats", {}).get("arms", {})
        lines.append(
            f"  [UNIFIED] AI:{u_ai.get('total',0)} trades | "
            f"{len(u_arms)} padrões | "
            f"LGBM:{len(u_lgbm)} amostras"
        )
        for suffix in ["m1", "bullex", "casatrader"]:
            broker = self._data.get("brokers", {}).get(suffix, {})
            ai = broker.get("ai_stats", {}).get("meta", {})
            lgbm = broker.get("lgbm_data", [])
            bt = broker.get("backtest_history", [])
            sess = broker.get("session", {})
            lines.append(
                f"  [{suffix}] AI:{ai.get('total',0)} trades | "
                f"LGBM:{len(lgbm)} amostras | "
                f"Backtest:{len(bt)} sinais | "
                f"Sessão: {sess.get('total',0)}T/{sess.get('wins',0)}W"
            )
        loss = self._data.get("loss_memory", [])
        lines.append(f"  Loss Memory: {len(loss)} registros")
        return "\n".join(lines)


# ========================= INSTÂNCIA GLOBAL =========================
# Para uso direto sem instanciar: from ws_data_manager import dm
dm = DataManager(auto_load=False)  # Lazy load — chamar dm.load() quando precisar
