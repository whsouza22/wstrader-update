"""
Loss Analyzer AI - Analisa motivos de LOSS para aprendizado

Quando um trade perde:
1. Captura o contexto completo (velas, S/R, estrutura, etc.)
2. Analisa o que deu errado
3. Categoriza o motivo do loss
4. Aprende padroes para evitar no futuro

Categorias de LOSS:
- CONTRA_TENDENCIA: Entrou contra a tendencia maior
- FALSO_BREAKOUT: S/R foi rompido e voltou
- VOLATILIDADE_ALTA: Mercado muito volatil/serrilhado
- TIMING_RUIM: Entrou no momento errado do pullback
- FORCA_INSUFICIENTE: Setup fraco, baixa confianca
- REVERSAO_RAPIDA: Mercado reverteu apos entrada
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import pandas as pd
import numpy as np


@dataclass
class LossContext:
    """Contexto completo de um trade perdedor"""
    timestamp: str
    ativo: str
    direcao: str  # CALL ou PUT
    preco_entrada: float
    preco_saida: float
    prejuizo: float

    # Contexto do mercado
    tendencia_m1: str  # ALTA, BAIXA, LATERAL
    tendencia_forca: float
    estrutura: str  # HH_HL, LH_LL, RANGE
    tinha_pullback: bool
    pullback_depth: float

    # Vela de entrada
    vela_tipo: str  # verde, vermelha, doji
    vela_tamanho_atr: float
    vela_pavio_sup: float
    vela_pavio_inf: float

    # S/R
    proximo_sr: str  # SUPORTE ou RESISTENCIA
    distancia_sr: float
    sr_foi_rompido: bool

    # Volatilidade
    atr_valor: float
    volatilidade: str  # baixa, normal, alta

    # Agentes
    agentes_votos: Dict[str, str]  # {agente: voto}
    consenso: int  # quantos agentes concordaram

    # Analise pos-loss
    categoria_loss: str = ""
    motivo_detalhado: str = ""
    licao_aprendida: str = ""


class LossAnalyzerAI:
    """
    IA que analisa cada LOSS para entender o motivo.

    Filosofia:
    - Cada loss e uma oportunidade de aprendizado
    - Categoriza os motivos para identificar padroes
    - Ajusta comportamento baseado nos padroes encontrados
    """

    CATEGORIES = [
        "CONTRA_TENDENCIA",
        "PADRAO_FRACO",
        "VOLATILIDADE_ALTA",
        "FORCA_INSUFICIENTE",
        "REVERSAO_RAPIDA",
        "FALSO_BREAKOUT",
        "TIMING_RUIM",
        "SR_INVALIDO",
        "MERCADO_IMPREVISIVEL",
        "DESCONHECIDO"
    ]

    def __init__(self, history_file: str = "loss_history.json"):
        self.history_file = history_file
        self.losses: List[LossContext] = []
        self.category_counts: Dict[str, int] = defaultdict(int)
        self.lessons_learned: List[str] = []

        self._load_history()

    def analyze_loss(
        self,
        df_m1: pd.DataFrame,
        direcao: str,
        preco_entrada: float,
        preco_saida: float,
        prejuizo: float,
        agentes_votos: Dict[str, str],
        atr_val: float,
        sr_info: Optional[Dict] = None,
        estrutura_info: Optional[Dict] = None,
        ativo: str = "UNKNOWN"
    ) -> LossContext:
        """
        Analisa um trade perdedor e categoriza o motivo.

        Returns:
            LossContext com analise completa
        """
        # Extrai informacoes das velas
        last_candle = df_m1.iloc[-1]
        c, o = last_candle["close"], last_candle["open"]
        h, l = last_candle["high"], last_candle["low"]

        body = abs(c - o)
        total_range = h - l if h > l else 0.0001

        vela_tipo = "verde" if c > o else "vermelha" if c < o else "doji"
        vela_tamanho_atr = total_range / atr_val if atr_val > 0 else 1.0
        vela_pavio_sup = (h - max(c, o)) / total_range if total_range > 0 else 0
        vela_pavio_inf = (min(c, o) - l) / total_range if total_range > 0 else 0

        # Analisa tendencia
        tendencia_m1, tendencia_forca = self._analyze_trend(df_m1)

        # Analisa estrutura
        estrutura = "RANGE"
        tinha_pullback = False
        pullback_depth = 0.0

        if estrutura_info:
            estrutura = estrutura_info.get("structure", "RANGE")
            tinha_pullback = estrutura_info.get("is_pullback", False)
            pullback_depth = estrutura_info.get("pullback_depth", 0.0)

        # Analisa S/R
        proximo_sr = "NONE"
        distancia_sr = 0.0
        sr_foi_rompido = False

        if sr_info:
            proximo_sr = sr_info.get("tipo", "NONE")
            distancia_sr = sr_info.get("distancia", 0.0)
            sr_foi_rompido = sr_info.get("rompido", False)

        # Volatilidade
        volatilidade = "normal"
        if atr_val > 0.002:
            volatilidade = "alta"
        elif atr_val < 0.0005:
            volatilidade = "baixa"

        # Conta consenso
        consenso = sum(1 for v in agentes_votos.values() if v == direcao)

        # Cria contexto
        context = LossContext(
            timestamp=datetime.now().isoformat(),
            ativo=ativo,
            direcao=direcao,
            preco_entrada=preco_entrada,
            preco_saida=preco_saida,
            prejuizo=prejuizo,
            tendencia_m1=tendencia_m1,
            tendencia_forca=tendencia_forca,
            estrutura=estrutura,
            tinha_pullback=tinha_pullback,
            pullback_depth=pullback_depth,
            vela_tipo=vela_tipo,
            vela_tamanho_atr=vela_tamanho_atr,
            vela_pavio_sup=vela_pavio_sup,
            vela_pavio_inf=vela_pavio_inf,
            proximo_sr=proximo_sr,
            distancia_sr=distancia_sr,
            sr_foi_rompido=sr_foi_rompido,
            atr_valor=atr_val,
            volatilidade=volatilidade,
            agentes_votos=agentes_votos,
            consenso=consenso
        )

        # === CATEGORIZA O LOSS ===
        context = self._categorize_loss(context)

        # Salva no historico
        self.losses.append(context)
        self.category_counts[context.categoria_loss] += 1
        self._save_history()

        return context

    def _analyze_trend(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Analisa tendencia das ultimas velas."""
        if len(df) < 10:
            return "LATERAL", 0.5

        ultimas = df.tail(10)
        verdes = sum(1 for _, r in ultimas.iterrows() if r["close"] > r["open"])

        # Variacao percentual
        var = (ultimas.iloc[-1]["close"] - ultimas.iloc[0]["close"]) / ultimas.iloc[0]["close"]

        # Detecta topos e fundos simples
        highs = [r["high"] for _, r in ultimas.iterrows()]
        lows = [r["low"] for _, r in ultimas.iterrows()]

        # Verifica se topos estao subindo ou descendo
        topos_subindo = highs[-1] > highs[0] and highs[-3] > highs[3]
        topos_descendo = highs[-1] < highs[0] and highs[-3] < highs[3]

        # ALTA: 6+ velas verdes OU subiu >0.1% OU topos subindo
        if verdes >= 6 or var > 0.001 or topos_subindo:
            forca = min(1.0, 0.5 + (verdes * 0.05) + (abs(var) * 50))
            return "ALTA", forca

        # BAIXA: 6+ velas vermelhas OU caiu >0.1% OU topos descendo
        if verdes <= 4 or var < -0.001 or topos_descendo:
            forca = min(1.0, 0.5 + ((10 - verdes) * 0.05) + (abs(var) * 50))
            return "BAIXA", forca

        return "LATERAL", 0.5

    def _categorize_loss(self, ctx: LossContext) -> LossContext:
        """Categoriza o motivo do loss com analise detalhada."""

        # Coleta informacoes para analise
        motivos_detectados = []

        # 1. CONTRA_TENDENCIA - Mais importante, verifica primeiro
        contra_tendencia = False
        if ctx.direcao == "CALL" and ctx.tendencia_m1 == "BAIXA":
            contra_tendencia = True
            motivos_detectados.append(f"CALL contra tendencia BAIXA (forca={ctx.tendencia_forca:.2f})")

        if ctx.direcao == "PUT" and ctx.tendencia_m1 == "ALTA":
            contra_tendencia = True
            motivos_detectados.append(f"PUT contra tendencia ALTA (forca={ctx.tendencia_forca:.2f})")

        if contra_tendencia and ctx.tendencia_forca > 0.55:
            ctx.categoria_loss = "CONTRA_TENDENCIA"
            ctx.motivo_detalhado = f"Entrada {ctx.direcao} contra tendencia {ctx.tendencia_m1}"
            ctx.licao_aprendida = "Operar somente NA DIRECAO da tendencia. LH+LL=PUT, HH+HL=CALL."
            return ctx

        # 2. VOLATILIDADE - Vela muito grande ou ATR alto
        if ctx.vela_tamanho_atr > 1.8:
            ctx.categoria_loss = "VOLATILIDADE_ALTA"
            ctx.motivo_detalhado = f"Vela muito grande ({ctx.vela_tamanho_atr:.1f}x ATR) - mercado instavel"
            ctx.licao_aprendida = "Evitar velas > 1.5x ATR. Mercado muito volatil para operar."
            return ctx

        # 3. VELA_CONTRA - Vela de entrada era contra a direcao
        vela_contra = False
        if ctx.direcao == "CALL" and ctx.vela_tipo == "vermelha":
            vela_contra = True
        if ctx.direcao == "PUT" and ctx.vela_tipo == "verde":
            vela_contra = True

        if vela_contra:
            ctx.categoria_loss = "PADRAO_FRACO"
            ctx.motivo_detalhado = f"Entrou {ctx.direcao} mas vela era {ctx.vela_tipo}"
            ctx.licao_aprendida = "Esperar vela de confirmacao na direcao antes de entrar."
            return ctx

        # 4. PAVIOS GRANDES - Indecisao no mercado
        if ctx.vela_pavio_sup > 0.35 and ctx.vela_pavio_inf > 0.35:
            ctx.categoria_loss = "PADRAO_FRACO"
            ctx.motivo_detalhado = f"Vela com pavios grandes (sup={ctx.vela_pavio_sup:.0%}, inf={ctx.vela_pavio_inf:.0%})"
            ctx.licao_aprendida = "Evitar velas com pavios > 35% - indicam indecisao."
            return ctx

        # 5. PAVIO CONTRA - Pavio grande na direcao contraria
        if ctx.direcao == "CALL" and ctx.vela_pavio_sup > 0.45:
            ctx.categoria_loss = "PADRAO_FRACO"
            ctx.motivo_detalhado = f"CALL mas pavio superior grande ({ctx.vela_pavio_sup:.0%}) - rejeicao de alta"
            ctx.licao_aprendida = "Pavio superior grande indica rejeicao. Nao entrar CALL."
            return ctx

        if ctx.direcao == "PUT" and ctx.vela_pavio_inf > 0.45:
            ctx.categoria_loss = "PADRAO_FRACO"
            ctx.motivo_detalhado = f"PUT mas pavio inferior grande ({ctx.vela_pavio_inf:.0%}) - rejeicao de baixa"
            ctx.licao_aprendida = "Pavio inferior grande indica rejeicao. Nao entrar PUT."
            return ctx

        # 6. CONSENSO FRACO (agora sao 5 agentes)
        if ctx.consenso < 4:
            ctx.categoria_loss = "FORCA_INSUFICIENTE"
            ctx.motivo_detalhado = f"Apenas {ctx.consenso}/5 agentes concordaram"
            ctx.licao_aprendida = "Exigir consenso de 4+ agentes para maior probabilidade."
            return ctx

        # 7. MERCADO LATERAL
        if ctx.tendencia_m1 == "LATERAL":
            ctx.categoria_loss = "REVERSAO_RAPIDA"
            ctx.motivo_detalhado = "Mercado sem direcao clara - movimentos aleatorios"
            ctx.licao_aprendida = "Evitar mercado lateral. Esperar formacao de tendencia."
            return ctx

        # 8. DOJI ou vela muito pequena
        if ctx.vela_tipo == "doji":
            ctx.categoria_loss = "PADRAO_FRACO"
            ctx.motivo_detalhado = "Vela doji - sem direcao definida"
            ctx.licao_aprendida = "Doji indica indecisao. Esperar proxima vela confirmar."
            return ctx

        # 9. S/R ROMPIDO
        if ctx.sr_foi_rompido:
            ctx.categoria_loss = "FALSO_BREAKOUT"
            ctx.motivo_detalhado = f"S/R ({ctx.proximo_sr}) foi rompido apos entrada"
            ctx.licao_aprendida = "S/R pode romper. Usar stop ou aguardar confirmacao."
            return ctx

        # 10. Se nenhum motivo claro - analisa combinacao
        if contra_tendencia:
            ctx.categoria_loss = "CONTRA_TENDENCIA"
            ctx.motivo_detalhado = f"Entrada contra tendencia (forca={ctx.tendencia_forca:.2f})"
            ctx.licao_aprendida = "Mesmo tendencia fraca, operar na direcao e mais seguro."
            return ctx

        # DESCONHECIDO - mercado imprevisivel
        ctx.categoria_loss = "MERCADO_IMPREVISIVEL"
        ctx.motivo_detalhado = "Setup parecia correto mas mercado foi contra - acontece"
        ctx.licao_aprendida = "Nem todo loss e erro. Gestao de risco protege nesses casos."
        return ctx

    def get_statistics(self) -> Dict:
        """Retorna estatisticas dos losses."""
        total = len(self.losses)
        if total == 0:
            return {"total": 0, "categories": {}}

        stats = {
            "total": total,
            "categories": {},
            "top_category": "",
            "avg_prejuizo": 0.0,
            "recent_pattern": "",
            "recommendation": ""
        }

        # Contagem por categoria
        for cat in self.CATEGORIES:
            count = self.category_counts.get(cat, 0)
            stats["categories"][cat] = {
                "count": count,
                "percent": (count / total) * 100
            }

        # Categoria mais comum
        if self.category_counts:
            top_cat = max(self.category_counts, key=self.category_counts.get)
            stats["top_category"] = top_cat
            stats["recommendation"] = self._get_recommendation(top_cat)

        # Media de prejuizo
        if self.losses:
            stats["avg_prejuizo"] = sum(l.prejuizo for l in self.losses) / total

        # Padrao recente (ultimos 5 losses)
        recent = self.losses[-5:] if len(self.losses) >= 5 else self.losses
        recent_cats = [l.categoria_loss for l in recent]
        if recent_cats:
            most_common = max(set(recent_cats), key=recent_cats.count)
            stats["recent_pattern"] = most_common

        return stats

    def _get_recommendation(self, category: str) -> str:
        """Retorna recomendacao baseada na categoria mais comum."""
        recommendations = {
            "CONTRA_TENDENCIA": "SO operar na DIRECAO da tendencia. LH+LL=PUT, HH+HL=CALL. Nunca contra.",
            "PADRAO_FRACO": "Esperar vela de confirmacao. Evitar dojis e velas com pavios > 35%.",
            "VOLATILIDADE_ALTA": "Filtrar velas > 1.5x ATR. Mercado muito volatil e imprevisivel.",
            "FORCA_INSUFICIENTE": "Exigir consenso de 4+/5 agentes para maior probabilidade.",
            "REVERSAO_RAPIDA": "Evitar mercado lateral. Esperar formacao de topos/fundos claros.",
            "FALSO_BREAKOUT": "Aguardar confirmacao de S/R. Usar vela de confirmacao antes de entrar.",
            "TIMING_RUIM": "Pullback ideal 38-62%. Muito fundo pode ser reversao.",
            "SR_INVALIDO": "Melhorar deteccao de S/R. Exigir mais toques ou ATR correto.",
            "MERCADO_IMPREVISIVEL": "Setup estava correto. Gestao de risco protege nesses casos."
        }
        return recommendations.get(category, "Revisar estrategia geral.")

    def get_lessons(self) -> List[str]:
        """Retorna licoes aprendidas dos ultimos losses."""
        if not self.losses:
            return []

        # Pega licoes unicas dos ultimos 10 losses
        recent = self.losses[-10:]
        lessons = list(set(l.licao_aprendida for l in recent if l.licao_aprendida))
        return lessons

    def should_trade(self, context: Dict) -> Tuple[bool, str]:
        """
        Verifica se deve operar baseado no aprendizado.

        Usa as licoes dos losses anteriores para filtrar trades.
        """
        stats = self.get_statistics()

        # Se top category e CONTRA_TENDENCIA e contexto mostra tendencia contraria
        if stats.get("top_category") == "CONTRA_TENDENCIA":
            if context.get("direcao") != context.get("tendencia_direcao"):
                return False, "Aprendizado: evitar contra tendencia"

        # Se top category e VOLATILIDADE_ALTA
        if stats.get("top_category") == "VOLATILIDADE_ALTA":
            if context.get("volatilidade") == "alta":
                return False, "Aprendizado: evitar alta volatilidade"

        # Se top category e FORCA_INSUFICIENTE
        if stats.get("top_category") == "FORCA_INSUFICIENTE":
            if context.get("consenso", 0) < 4:
                return False, "Aprendizado: exigir consenso total"

        return True, "OK"

    def _load_history(self):
        """Carrega historico de losses."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)

                self.category_counts = defaultdict(int, data.get("category_counts", {}))

                # Carrega losses (simplificado)
                for loss_data in data.get("losses", [])[-100:]:  # Ultimos 100
                    try:
                        ctx = LossContext(**loss_data)
                        self.losses.append(ctx)
                    except:
                        pass

                print(f"[LOSS-AI] Historico carregado: {len(self.losses)} losses")

            except Exception as e:
                print(f"[LOSS-AI] Erro ao carregar historico: {e}")

    def _save_history(self):
        """Salva historico de losses."""
        try:
            data = {
                "category_counts": dict(self.category_counts),
                "losses": [asdict(l) for l in self.losses[-100:]],  # Ultimos 100
                "last_updated": datetime.now().isoformat()
            }

            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"[LOSS-AI] Erro ao salvar: {e}")

    def print_report(self):
        """Imprime relatorio de analise."""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("           RELATORIO DE ANALISE DE LOSSES")
        print("=" * 60)
        print(f"\nTotal de losses analisados: {stats['total']}")

        if stats['total'] > 0:
            print(f"Prejuizo medio: ${stats['avg_prejuizo']:.2f}")
            print(f"\nCategoria mais comum: {stats['top_category']}")
            print(f"Recomendacao: {stats['recommendation']}")

            print("\n--- Distribuicao por Categoria ---")
            for cat, info in sorted(stats['categories'].items(), key=lambda x: -x[1]['count']):
                if info['count'] > 0:
                    bar = "#" * int(info['percent'] / 5)
                    print(f"{cat:20} | {info['count']:3} ({info['percent']:5.1f}%) {bar}")

            print("\n--- Licoes Recentes ---")
            for i, lesson in enumerate(self.get_lessons()[:5], 1):
                print(f"{i}. {lesson}")

        print("\n" + "=" * 60)


# ============================================================================
#                         INSTANCIA GLOBAL
# ============================================================================

_loss_analyzer: Optional[LossAnalyzerAI] = None


def get_loss_analyzer() -> LossAnalyzerAI:
    """Retorna instancia global do analisador."""
    global _loss_analyzer
    if _loss_analyzer is None:
        _loss_analyzer = LossAnalyzerAI()
    return _loss_analyzer


# ============================================================================
#                         TESTE LOCAL
# ============================================================================

if __name__ == "__main__":
    analyzer = get_loss_analyzer()

    # Simula alguns losses
    print("Simulando losses para teste...")

    # Loss 1: Contra tendencia
    analyzer.analyze_loss(
        df_m1=pd.DataFrame([
            {"open": 1.17, "high": 1.171, "low": 1.168, "close": 1.169},
            {"open": 1.169, "high": 1.170, "low": 1.167, "close": 1.168},
            {"open": 1.168, "high": 1.169, "low": 1.166, "close": 1.167},
        ] * 10),
        direcao="CALL",
        preco_entrada=1.167,
        preco_saida=1.165,
        prejuizo=-10.0,
        agentes_votos={"SR": "CALL", "Tendencia": "PUT", "Padroes": "CALL", "CNN": "CALL"},
        atr_val=0.0015,
        ativo="EURUSD-OTC"
    )

    # Loss 2: Volatilidade alta
    analyzer.analyze_loss(
        df_m1=pd.DataFrame([
            {"open": 1.17, "high": 1.175, "low": 1.165, "close": 1.172},
        ] * 10),
        direcao="PUT",
        preco_entrada=1.172,
        preco_saida=1.175,
        prejuizo=-10.0,
        agentes_votos={"SR": "PUT", "Tendencia": "PUT", "Padroes": "PUT", "CNN": "PUT"},
        atr_val=0.0035,
        ativo="EURUSD-OTC"
    )

    # Loss 3: Forca insuficiente
    analyzer.analyze_loss(
        df_m1=pd.DataFrame([
            {"open": 1.17, "high": 1.171, "low": 1.169, "close": 1.1705},
        ] * 10),
        direcao="CALL",
        preco_entrada=1.1705,
        preco_saida=1.169,
        prejuizo=-10.0,
        agentes_votos={"SR": "CALL", "Tendencia": "NEUTRO", "Padroes": "NEUTRO", "CNN": "CALL"},
        atr_val=0.0010,
        ativo="EURUSD-OTC"
    )

    # Imprime relatorio
    analyzer.print_report()
