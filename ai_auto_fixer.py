"""
AI Auto-Fixer - Sistema de IA que APRENDE com os LOSS

SEM REGRAS FIXAS - A IA comeca permitindo TUDO e vai aprendendo
com os erros para evitar repetir no futuro.

Fluxo:
1. Entrada: IA analisa mas NAO BLOQUEIA (apenas observa no inicio)
2. LOSS: IA analisa o motivo do erro e APRENDE
3. Proxima entrada similar: IA pode bloquear SE aprendeu com varios LOSS

Autor: AI Auto System
"""

import os
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Tenta importar OpenAI
AI_AVAILABLE = False
openai_client = None
try:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.startswith("sk-"):
        openai_client = OpenAI(api_key=api_key)
        AI_AVAILABLE = True
except ImportError:
    pass

class AIAutoFixer:
    """
    Sistema de IA que APRENDE com LOSS - SEM REGRAS FIXAS.
    
    Funciona assim:
    - Comeca permitindo TODAS as entradas
    - Quando ocorre LOSS, analisa o motivo com IA
    - Cria regras DINAMICAS baseado nos padroes de LOSS
    - So bloqueia quando tem CERTEZA (varios LOSS iguais)
    """
    
    def __init__(self, auto_apply: bool = False):
        self.auto_apply = auto_apply
        self.history_file = "ai_fixer_learning.json"
        
        # Contadores
        self.total_analyzed = 0
        self.total_blocked = 0
        self.correct_blocks = 0
        
        # Padroes de LOSS aprendidos (a IA preenche)
        # Formato: {"EURUSD_CALL_engolfo_alta_DOWN": [loss1, loss2, ...]}
        self.loss_patterns: Dict[str, List[Dict]] = defaultdict(list)
        
        # Padroes de WIN aprendidos
        self.win_patterns: Dict[str, List[Dict]] = defaultdict(list)
        
        # Regras aprendidas pela IA (COMECA VAZIO!)
        self.learned_rules: List[Dict] = []
        
        # Minimo de LOSS iguais para criar regra de bloqueio
        self.min_losses_to_block = 3
        
        # Carrega historico de aprendizado
        self._load_history()
        
        print(f"[AI-FIXER] Inicializado | OpenAI: {'SIM' if AI_AVAILABLE else 'NAO'} | Regras aprendidas: {len(self.learned_rules)}")
    
    def _load_history(self):
        """Carrega historico de aprendizado."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.loss_patterns = defaultdict(list, data.get("loss_patterns", {}))
                    self.win_patterns = defaultdict(list, data.get("win_patterns", {}))
                    self.learned_rules = data.get("learned_rules", [])
                    self.total_analyzed = data.get("total_analyzed", 0)
                    self.total_blocked = data.get("total_blocked", 0)
                    self.correct_blocks = data.get("correct_blocks", 0)
                    print(f"[AI-FIXER] Historico carregado: {len(self.learned_rules)} regras, {sum(len(v) for v in self.loss_patterns.values())} losses")
            except Exception as e:
                print(f"[AI-FIXER] Erro ao carregar historico: {e}")
    
    def _save_history(self):
        """Salva historico de aprendizado."""
        try:
            data = {
                "loss_patterns": dict(self.loss_patterns),
                "win_patterns": dict(self.win_patterns),
                "learned_rules": self.learned_rules,
                "total_analyzed": self.total_analyzed,
                "total_blocked": self.total_blocked,
                "correct_blocks": self.correct_blocks,
                "last_updated": datetime.now().isoformat()
            }
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[AI-FIXER] Erro ao salvar historico: {e}")
    
    def _create_pattern_key(self, context: Dict) -> str:
        """Cria chave unica para o padrao - usa MICRO TENDENCIA."""
        ativo = context.get("ativo", "UNK")
        direction = context.get("direction", "UNK")
        pattern = context.get("pattern", "none")
        # Prefere micro_trend se disponivel, senao usa trend
        micro = context.get("micro_trend", context.get("trend", "UNK"))
        
        # Chave: ATIVO_DIRECAO_PATTERN_MICROTREND
        return f"{ativo}_{direction}_{pattern}_{micro}"
    
    def should_enter(self, context: Dict) -> Tuple[bool, str]:
        """
        Decide se deve entrar na operacao.
        
        IMPORTANTE: Comeca permitindo TUDO!
        So bloqueia quando aprendeu com LOSS suficientes.
        
        Returns:
            (should_enter: bool, reason: str)
        """
        self.total_analyzed += 1
        
        pattern_key = self._create_pattern_key(context)
        
        # Verifica se tem regra aprendida para este padrao
        for rule in self.learned_rules:
            if self._matches_rule(context, rule):
                # So bloqueia se a regra diz BLOCK
                if rule.get("action") == "BLOCK":
                    self.total_blocked += 1
                    reason = rule.get("reason", "Padrao bloqueado por aprendizado")
                    loss_count = rule.get("loss_count", 0)
                    return False, f"APRENDIDO: {reason} ({loss_count} losses anteriores)"
        
        # Verifica se tem muitos LOSS com este padrao especifico
        loss_list = self.loss_patterns.get(pattern_key, [])
        win_list = self.win_patterns.get(pattern_key, [])
        
        # So bloqueia se:
        # 1. Tem pelo menos min_losses_to_block LOSS
        # 2. Winrate do padrao e menor que 40%
        if len(loss_list) >= self.min_losses_to_block:
            total = len(loss_list) + len(win_list)
            winrate = len(win_list) / total if total > 0 else 0
            
            if winrate < 0.40:
                self.total_blocked += 1
                return False, f"APRENDIDO: Padrao {pattern_key} tem {len(loss_list)} losses (winrate {winrate:.0%})"
        
        # PERMITE A ENTRADA (padrao)
        return True, "OK - Sem restricao aprendida"
    
    def _matches_rule(self, context: Dict, rule: Dict) -> bool:
        """Verifica se contexto bate com uma regra aprendida."""
        # Verifica ativo (ou * para qualquer)
        rule_ativo = rule.get("ativo", "*")
        if rule_ativo != "*" and rule_ativo != context.get("ativo"):
            return False
        
        # Verifica direcao
        rule_dir = rule.get("direction", "*")
        if rule_dir != "*" and rule_dir != context.get("direction"):
            return False
        
        # Verifica padrao
        rule_pattern = rule.get("pattern", "*")
        if rule_pattern != "*" and rule_pattern != context.get("pattern"):
            return False
        
        # Verifica tendencia
        rule_trend = rule.get("trend", "*")
        if rule_trend != "*" and rule_trend != context.get("trend"):
            return False
        
        return True
    
    def analyze_loss(self, trade_info: Dict, candles_df: pd.DataFrame) -> Dict:
        """
        APOS LOSS: Analisa o motivo do erro e APRENDE.
        
        Esta e a funcao principal de aprendizado!
        """
        pattern_key = self._create_pattern_key(trade_info)
        
        # Extrai contexto das velas
        candle_context = self._extract_candle_context(candles_df)
        
        # Combina informacoes
        full_context = {**trade_info, **candle_context, "timestamp": datetime.now().isoformat()}
        
        # Salva no historico de LOSS
        self.loss_patterns[pattern_key].append(full_context)
        
        # Usa IA para analisar o motivo (se disponivel)
        if AI_AVAILABLE and openai_client:
            analysis = self._ai_analyze_loss(trade_info, candle_context)
        else:
            analysis = self._simple_analyze_loss(trade_info, candle_context)
        
        # Verifica se deve criar regra de bloqueio
        loss_list = self.loss_patterns[pattern_key]
        if len(loss_list) >= self.min_losses_to_block:
            self._maybe_create_rule(pattern_key, loss_list, analysis)
        
        # Salva historico
        self._save_history()
        
        return analysis
    
    def _ai_analyze_loss(self, trade_info: Dict, candle_context: Dict) -> Dict:
        """Usa OpenAI para analisar o LOSS."""
        try:
            prompt = f"""Analise este trade que deu LOSS e descubra o motivo do erro:

TRADE:
- Ativo: {trade_info.get('ativo')}
- Direcao: {trade_info.get('direction')}
- Padrao usado: {trade_info.get('pattern')}  
- Probabilidade ML: {trade_info.get('ml_prob', 0):.2%}
- Score: {trade_info.get('score', 0):.2f}
- Horario: {trade_info.get('hour')}h
- Resultado: {trade_info.get('pnl', 0):.2f}

CONTEXTO DO MERCADO:
- Tendencia: {candle_context.get('trend', 'N/A')}
- Volatilidade: {candle_context.get('volatility_ratio', 1):.2f}x normal
- Ultimas 5 velas: {candle_context.get('last_5_colors', 'N/A')}
- Range das velas: {candle_context.get('range_pips', 0):.1f} pips

Por que este trade falhou? Responda em JSON:
{{
  "cause": "explicacao curta do motivo",
  "category": "CONTRA_TENDENCIA|VOLATILIDADE|PADRAO_FRACO|TIMING|RANGE_LATERAL|OUTRO",
  "confidence": 0.0-1.0,
  "fix_suggestion": "o que fazer para evitar este erro",
  "should_block_pattern": true/false
}}"""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Voce e um analista de trading que identifica erros em operacoes. Seja direto e objetivo. Responda APENAS em JSON valido."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            
            # Extrai JSON da resposta
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            
            return {"cause": content, "category": "OUTRO", "fix_suggestion": "Analisar manualmente"}
            
        except Exception as e:
            print(f"[AI-FIXER] Erro na analise IA: {e}")
            return self._simple_analyze_loss(trade_info, candle_context)
    
    def _simple_analyze_loss(self, trade_info: Dict, candle_context: Dict) -> Dict:
        """Analise simples sem IA."""
        direction = trade_info.get("direction", "")
        trend = candle_context.get("trend", "RANGE")
        colors = candle_context.get("last_5_colors", "")
        vol_ratio = candle_context.get("volatility_ratio", 1.0)
        
        cause = "Motivo nao identificado claramente"
        category = "OUTRO"
        fix = "Continuar observando padroes similares"
        should_block = False
        
        # Analisa possiveis causas (SEM BLOQUEAR, apenas identifica)
        if direction == "CALL" and trend == "DOWN":
            cause = "Entrou CALL durante tendencia de baixa"
            category = "CONTRA_TENDENCIA"
            fix = "Observar: CALL em tendencia de baixa pode ser arriscado"
            should_block = True
        elif direction == "PUT" and trend == "UP":
            cause = "Entrou PUT durante tendencia de alta"
            category = "CONTRA_TENDENCIA"
            fix = "Observar: PUT em tendencia de alta pode ser arriscado"
            should_block = True
        elif vol_ratio > 2.0:
            cause = f"Volatilidade muito alta ({vol_ratio:.1f}x normal)"
            category = "VOLATILIDADE"
            fix = "Observar: Alta volatilidade aumenta risco"
            should_block = True
        elif "GGGGG" in colors or "RRRRR" in colors:
            cause = "Mercado em movimento forte sem reversao"
            category = "TIMING"
            fix = "Observar: Nao entrar contra movimento forte"
        elif vol_ratio < 0.3:
            cause = "Mercado muito parado (baixa volatilidade)"
            category = "RANGE_LATERAL"
            fix = "Observar: Mercado lateral dificulta previsao"
        
        return {
            "cause": cause,
            "category": category,
            "fix_suggestion": fix,
            "confidence": 0.6,
            "should_block_pattern": should_block
        }
    
    def _extract_candle_context(self, df: pd.DataFrame) -> Dict:
        """Extrai contexto das velas para analise - MICRO TENDENCIA + REGIÃO."""
        if df is None or len(df) < 3:
            return {"trend": "UNKNOWN", "volatility_ratio": 1.0, "last_5_colors": "", "micro_trend": "UNKNOWN", "region": "NEUTRAL"}
        
        try:
            # Ultimas 8 velas para análise de região (ou menos se nao tiver)
            n_colors = min(8, len(df))
            last_n = df.tail(n_colors)
            colors = ""
            greens = 0
            reds = 0
            for _, row in last_n.iterrows():
                if row.get("close", 0) > row.get("open", 0):
                    colors += "G"
                    greens += 1
                else:
                    colors += "R"
                    reds += 1
            
            # MICRO TENDENCIA: Ultimas 3 velas apenas
            if len(df) >= 3:
                first_close = df.iloc[-3]["close"]
                last_close = df.iloc[-1]["close"]
                diff_pct = (last_close - first_close) / first_close if first_close else 0
                
                # Threshold menor para micro tendencia (5 pips = 0.0005)
                if diff_pct > 0.0005:
                    micro_trend = "UP"
                elif diff_pct < -0.0005:
                    micro_trend = "DOWN"
                else:
                    micro_trend = "RANGE"
                
                # Confirma com cores das ultimas 3
                last_3_colors = colors[-3:] if len(colors) >= 3 else colors
                if last_3_colors.count("G") >= 2 and micro_trend != "DOWN":
                    micro_trend = "UP"
                elif last_3_colors.count("R") >= 2 and micro_trend != "UP":
                    micro_trend = "DOWN"
            else:
                micro_trend = "RANGE"
            
            # Tendencia geral (5 velas)
            if len(df) >= 5:
                first_close_5 = df.iloc[-5]["close"]
                last_close_5 = df.iloc[-1]["close"]
                diff_5 = (last_close_5 - first_close_5) / first_close_5 if first_close_5 else 0
                
                if diff_5 > 0.001:
                    trend = "UP"
                elif diff_5 < -0.001:
                    trend = "DOWN"
                else:
                    trend = "RANGE"
            else:
                trend = micro_trend
            
            # Volatilidade
            if "high" in df.columns and "low" in df.columns:
                ranges = df["high"] - df["low"]
                current_range = ranges.iloc[-1]
                avg_range = ranges.mean()
                vol_ratio = current_range / avg_range if avg_range > 0 else 1.0
            else:
                vol_ratio = 1.0
            
            # 🎯 DETECÇÃO AGRESSIVA DE REGIÃO (TOPO vs FUNDO)
            # Analisa posição atual vs range das últimas 10-20 velas
            region = "NEUTRAL"
            if len(df) >= 8:
                recent = df.tail(10) if len(df) >= 10 else df
                high_max = recent["high"].max()
                low_min = recent["low"].min()
                current_close = df.iloc[-1]["close"]
                range_total = high_max - low_min
                
                if range_total > 0:
                    position = (current_close - low_min) / range_total
                    
                    # TOPO: preço > 70% do range OU 6+ velas verdes
                    if position > 0.70 or greens >= 6:
                        region = "TOPO"
                    # FUNDO: preço < 30% do range OU 6+ velas vermelhas
                    elif position < 0.30 or reds >= 6:
                        region = "FUNDO"
                    # Verdes/vermelhos dominantes
                    elif greens >= 5 and position > 0.55:
                        region = "TOPO"
                    elif reds >= 5 and position < 0.45:
                        region = "FUNDO"
            
            return {
                "trend": trend,
                "micro_trend": micro_trend,
                "volatility_ratio": vol_ratio,
                "last_5_colors": colors,
                "greens": greens,
                "reds": reds,
                "region": region,
                "range_pips": (df.iloc[-1].get("high", 0) - df.iloc[-1].get("low", 0)) * 10000
            }
        except Exception as e:
            return {"trend": "UNKNOWN", "micro_trend": "UNKNOWN", "volatility_ratio": 1.0, "last_5_colors": "", "region": "NEUTRAL"}
    
    def _maybe_create_rule(self, pattern_key: str, loss_list: List[Dict], analysis: Dict):
        """Cria regra de bloqueio se identificou padrao recorrente."""
        # So cria regra se analise sugere bloqueio
        if not analysis.get("should_block_pattern", False):
            return
        
        # Extrai info do pattern_key
        parts = pattern_key.split("_")
        if len(parts) >= 4:
            ativo = parts[0]
            direction = parts[1]
            pattern = parts[2]
            trend = parts[3] if len(parts) > 3 else "*"
        else:
            return
        
        # Verifica se regra ja existe
        for existing in self.learned_rules:
            if (existing.get("ativo") == ativo and 
                existing.get("direction") == direction and
                existing.get("pattern") == pattern):
                # Atualiza contagem
                existing["loss_count"] = len(loss_list)
                existing["last_updated"] = datetime.now().isoformat()
                return
        
        # Cria nova regra
        new_rule = {
            "ativo": ativo,
            "direction": direction,
            "pattern": pattern,
            "trend": trend,
            "action": "BLOCK",
            "reason": analysis.get("cause", "Multiplos losses identificados"),
            "category": analysis.get("category", "OUTRO"),
            "loss_count": len(loss_list),
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        self.learned_rules.append(new_rule)
        print(f"[AI-FIXER] >>> NOVA REGRA APRENDIDA: {pattern_key} -> BLOCK ({len(loss_list)} losses)")
    
    def record_win(self, trade_info: Dict, candles_df: pd.DataFrame):
        """
        APOS WIN: Registra padrao positivo para aprendizado.
        """
        pattern_key = self._create_pattern_key(trade_info)
        
        candle_context = self._extract_candle_context(candles_df)
        full_context = {**trade_info, **candle_context, "timestamp": datetime.now().isoformat()}
        
        self.win_patterns[pattern_key].append(full_context)
        
        # Salva historico
        self._save_history()
    
    def get_stats(self) -> Dict:
        """Retorna estatisticas do sistema."""
        total_losses = sum(len(v) for v in self.loss_patterns.values())
        total_wins = sum(len(v) for v in self.win_patterns.values())
        
        return {
            "total_analyzed": self.total_analyzed,
            "total_blocked": self.total_blocked,
            "total_losses_learned": total_losses,
            "total_wins_learned": total_wins,
            "learned_rules": len(self.learned_rules),
            "rules": [r.get("reason", "N/A")[:50] for r in self.learned_rules[:5]],
            "ai_available": AI_AVAILABLE
        }
    
    def get_rules_summary(self) -> str:
        """Retorna resumo das regras aprendidas."""
        if not self.learned_rules:
            return "Nenhuma regra aprendida ainda. O sistema vai aprender com os LOSS."
        
        lines = ["=== REGRAS APRENDIDAS ==="]
        for rule in self.learned_rules:
            lines.append(f"- {rule.get('pattern')} {rule.get('direction')} em {rule.get('trend')}: {rule.get('reason')} ({rule.get('loss_count')} losses)")
        
        return "\n".join(lines)
    
    def reset_learning(self):
        """Reseta todo aprendizado (usar com cuidado!)."""
        self.loss_patterns = defaultdict(list)
        self.win_patterns = defaultdict(list)
        self.learned_rules = []
        self.total_analyzed = 0
        self.total_blocked = 0
        self._save_history()
        print("[AI-FIXER] Aprendizado resetado!")
    
    def analyze_entry_and_exp(self, context: Dict, candles_df: pd.DataFrame) -> Dict:
        """
        🚀 ANÁLISE ULTRA-RÁPIDA - Trader Profissional
        
        Prioridade: Velocidade > Precisão da IA
        Usa análise LOCAL rápida, só chama OpenAI se necessário.
        """
        # Extrai contexto das velas
        candle_ctx = self._extract_candle_context(candles_df)
        full_context = {**context, **candle_ctx}
        
        direction = context.get('direction', 'CALL')
        pattern = context.get('pattern', '')
        region = candle_ctx.get('region', 'NEUTRAL')
        trend = candle_ctx.get('trend', 'RANGE')
        micro = candle_ctx.get('micro_trend', 'RANGE')
        greens = candle_ctx.get('greens', 0)
        reds = candle_ctx.get('reds', 0)
        
        # 🛡️ BLOQUEIO POR REGIÃO - MAIS IMPORTANTE!
        # CALL no TOPO = BLOQUEADO (mercado esticado para cima)
        if direction == "CALL" and region == "TOPO":
            return {
                "should_enter": False,
                "direction": direction,
                "reason": f"❌ CALL bloqueado: região de TOPO ({greens} verdes)",
                "suggested_exp": 1,
                "confidence": 0.0,
                "trend": trend,
                "micro_trend": micro,
                "region": region
            }
        
        # PUT no FUNDO = BLOQUEADO (mercado esticado para baixo)
        if direction == "PUT" and region == "FUNDO":
            return {
                "should_enter": False,
                "direction": direction,
                "reason": f"❌ PUT bloqueado: região de FUNDO ({reds} vermelhas)",
                "suggested_exp": 1,
                "confidence": 0.0,
                "trend": trend,
                "micro_trend": micro,
                "region": region
            }
        
        # 🎯 PADRÕES DE REVERSÃO - Validação rápida
        reversal_patterns = ['engolfo_baixa', 'engolfo_alta', 'martelo', 'estrela_cad', 
                             'estrela_cadente', 'estrela_manha', 'estrela_noite',
                             'piercing', 'nuvem_negra', 'harami_alta', 'harami_baixa']
        is_reversal = any(p in pattern for p in reversal_patterns)
        
        # Padrão de ALTA no TOPO = deve ser PUT (reversão)
        alta_patterns = ['engolfo_alta', 'martelo', 'estrela_manha', 'piercing', 'harami_alta', '3_soldados']
        baixa_patterns = ['engolfo_baixa', 'estrela_cad', 'estrela_cadente', 'estrela_noite', 'nuvem_negra', 'harami_baixa', '3_corvos']
        
        # Verifica se padrão está na região CORRETA
        is_alta = any(p in pattern for p in alta_patterns)
        is_baixa = any(p in pattern for p in baixa_patterns)
        
        # ⚡ FAST PATH: Padrão + Região alinhados = entrada direta
        if is_baixa and region == "TOPO" and direction == "PUT":
            return {
                "should_enter": True,
                "direction": "PUT",
                "reason": f"✅ {pattern} no TOPO = PUT confirmado",
                "suggested_exp": 1,
                "confidence": 0.90,
                "trend": trend,
                "micro_trend": micro,
                "region": region
            }
        
        if is_alta and region == "FUNDO" and direction == "CALL":
            return {
                "should_enter": True,
                "direction": "CALL",
                "reason": f"✅ {pattern} no FUNDO = CALL confirmado",
                "suggested_exp": 1,
                "confidence": 0.90,
                "trend": trend,
                "micro_trend": micro,
                "region": region
            }
        
        # ⚠️ CONFLITO: Padrão de ALTA no TOPO = BLOQUEADO  
        if is_alta and region == "TOPO":
            return {
                "should_enter": False,
                "direction": direction,
                "reason": f"❌ {pattern} no TOPO = conflito (espera reversão)",
                "suggested_exp": 1,
                "confidence": 0.0,
                "trend": trend,
                "micro_trend": micro,
                "region": region
            }
        
        # ⚠️ CONFLITO: Padrão de BAIXA no FUNDO = BLOQUEADO
        if is_baixa and region == "FUNDO":
            return {
                "should_enter": False,
                "direction": direction,
                "reason": f"❌ {pattern} no FUNDO = conflito (espera reversão)",
                "suggested_exp": 1,
                "confidence": 0.0,
                "trend": trend,
                "micro_trend": micro,
                "region": region
            }
        
        # 🚀 FAST PATH para região NEUTRA com padrão forte
        if region == "NEUTRAL" and is_reversal:
            return {
                "should_enter": True,
                "direction": direction,
                "reason": f"✅ {pattern} confirmado (região neutra)",
                "suggested_exp": 1,
                "confidence": 0.80,
                "trend": trend,
                "micro_trend": micro,
                "region": region
            }
        
        # Sem padrão definido - usa análise simples de tendência
        return self._simple_analyze_entry_and_exp(full_context)
    
    def _ai_analyze_entry_and_exp(self, context: Dict) -> Dict:
        """Análise RÁPIDA como trader profissional - padrão + região + confirmação."""
        try:
            pattern = context.get('pattern', '')
            direction = context.get('direction', 'CALL')
            trend = context.get('trend', 'RANGE')
            micro = context.get('micro_trend', 'RANGE')
            colors = context.get('last_5_colors', '')
            score = context.get('score', 0.8)
            vol = context.get('volatility_ratio', 1.0)
            
            # PADRÕES DE REVERSÃO - direção FIXA
            reversal_patterns = ['engolfo_baixa', 'engolfo_alta', 'martelo', 'estrela_cad', 
                                 'estrela_cadente', 'estrela_manha', 'estrela_noite',
                                 'piercing', 'nuvem_negra', 'harami_alta', 'harami_baixa']
            is_reversal = any(p in pattern for p in reversal_patterns)
            
            # 🚀 PROMPT OTIMIZADO - Trader Profissional (máximo 100 tokens)
            prompt = f"""ANÁLISE RÁPIDA - Confirme entrada:
Ativo: {context.get('ativo')} | Dir: {direction} | Padrão: {pattern or 'nenhum'} | Score: {score:.0%}
Trend: {trend} | Micro: {micro} | Velas: {colors} | Vol: {vol:.1f}x

REGRAS TRADER PRO:
- Reversão ({pattern}): {direction} FIXO se padrão válido
- Região: Verde dominante=CALL, Vermelho=PUT
- Alinhamento: padrão+trend+micro na mesma direção=ENTRAR
- Conflito: trend≠micro≠padrão=NÃO ENTRAR

JSON: {{"enter":bool,"dir":"{direction}","exp":1-4,"conf":0-1}}"""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,  # ⚡ Reduzido para velocidade
                temperature=0.1  # ⚡ Mais determinístico = mais rápido
            )
            
            content = response.choices[0].message.content
            
            # Extrai JSON rápido
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                r = json.loads(json_match.group())
                
                # Mapeia campos simplificados
                result = {
                    "should_enter": r.get("enter", True),
                    "direction": r.get("dir", direction),
                    "suggested_exp": r.get("exp", 1),
                    "confidence": r.get("conf", 0.7),
                    "reason": "IA confirmou" if r.get("enter", True) else "IA bloqueou",
                    "trend": trend,
                    "micro_trend": micro
                }
                
                # 🛡️ PROTEÇÃO: Reversão = direção original
                if is_reversal and result["direction"] != direction:
                    result["direction"] = direction
                    result["reason"] = f"Padrão {pattern} confirma {direction}"
                
                return result
            
            # Fallback rápido
            return self._simple_analyze_entry_and_exp(context)
            
        except Exception as e:
            print(f"[AI-FIXER] Erro na análise IA: {e}")
            return self._simple_analyze_entry_and_exp(context)
    
    def _simple_analyze_entry_and_exp(self, context: Dict) -> Dict:
        """Análise simples sem IA - decide direção e expiração com REGIÃO."""
        direction = context.get("direction", "CALL")
        pattern = context.get("pattern", "")
        trend = context.get("trend", "RANGE")
        micro_trend = context.get("micro_trend", "RANGE")
        vol_ratio = context.get("volatility_ratio", 1.0)
        last_5_colors = context.get("last_5_colors", "")
        region = context.get("region", "NEUTRAL")
        
        should_enter = True
        final_direction = direction  # Padrão: mantém a direção original
        reason = "OK"
        suggested_exp = 1  # Padrão
        confidence = 0.7
        
        # 🛡️ PADRÕES DE REVERSÃO - NÃO mudar direção!
        reversal_patterns = [
            'engolfo_baixa', 'engolfo_alta', 
            'martelo', 'estrela_cad', 'estrela_cadente',
            'estrela_manha', 'estrela_noite',
            'piercing', 'nuvem_negra',
            'harami_alta', 'harami_baixa'
        ]
        is_reversal = any(p in pattern for p in reversal_patterns)
        
        # Conta cores das últimas 5 velas
        green_count = last_5_colors.count("G") if last_5_colors else 0
        red_count = last_5_colors.count("R") if last_5_colors else 0
        
        # 🎯 REGRA 0: BLOQUEIO POR REGIÃO (mais importante!)
        if region == "TOPO" and direction == "CALL":
            return {
                "should_enter": False,
                "direction": direction,
                "reason": f"❌ CALL bloqueado: região de TOPO",
                "suggested_exp": 1,
                "confidence": 0.0,
                "trend": trend,
                "micro_trend": micro_trend,
                "region": region
            }
        if region == "FUNDO" and direction == "PUT":
            return {
                "should_enter": False,
                "direction": direction,
                "reason": f"❌ PUT bloqueado: região de FUNDO",
                "suggested_exp": 1,
                "confidence": 0.0,
                "trend": trend,
                "micro_trend": micro_trend,
                "region": region
            }
        
        # REGRA 1: PADRÕES DE REVERSÃO - manter direção original!
        if is_reversal:
            final_direction = direction
            suggested_exp = 1
            confidence = 0.85
            reason = f"Padrão de reversão {pattern} - mantendo {direction}"
        # REGRA 2: Se micro_trend é claro E NÃO é reversão E região permite
        elif micro_trend == "UP" and region != "TOPO":
            final_direction = "CALL"
            suggested_exp = 1
            confidence = 0.8
            reason = f"Micro tendência UP - CALL"
        elif micro_trend == "DOWN" and region != "FUNDO":
            final_direction = "PUT"
            suggested_exp = 1
            confidence = 0.8
            reason = f"Micro tendência DOWN - PUT"
        
        # REGRA 3: Se tendência é clara e alinhada
        elif trend == "UP" and green_count >= 3:
            final_direction = "CALL"
            suggested_exp = 1
            confidence = 0.85
            reason = f"Tendência UP + {green_count} verdes - CALL"
        elif trend == "DOWN" and red_count >= 3:
            final_direction = "PUT"
            suggested_exp = 1
            confidence = 0.85
            reason = f"Tendência DOWN + {red_count} vermelhas - PUT"
        
        # REGRA 3: RANGE mas com cores dominantes - entra na direção predominante
        elif trend == "RANGE":
            if green_count >= 3 and red_count <= 1:
                final_direction = "CALL"
                suggested_exp = 2
                confidence = 0.65
                reason = f"RANGE mas {green_count} verdes - CALL"
            elif red_count >= 3 and green_count <= 1:
                final_direction = "PUT"
                suggested_exp = 2
                confidence = 0.65
                reason = f"RANGE mas {red_count} vermelhas - PUT"
            elif micro_trend != "RANGE":
                # Micro tendência existe, seguir ela
                final_direction = "CALL" if micro_trend == "UP" else "PUT"
                suggested_exp = 2
                confidence = 0.6
                reason = f"RANGE geral mas micro {micro_trend}"
            else:
                # RANGE total sem direção clara - NÃO ENTRAR
                should_enter = False
                reason = "RANGE total sem direção clara"
                confidence = 0.0
                return {
                    "should_enter": should_enter,
                    "direction": direction,
                    "reason": reason,
                    "suggested_exp": 1,
                    "confidence": confidence,
                    "trend": trend,
                    "micro_trend": micro_trend
                }
        
        # REGRA 4: Ajusta expiração por volatilidade
        if vol_ratio > 2.0:
            suggested_exp = min(4, suggested_exp + 2)
            reason += f" | alta vol ({vol_ratio:.1f}x)"
            confidence *= 0.9
        elif vol_ratio > 1.5:
            suggested_exp = min(4, suggested_exp + 1)
        elif vol_ratio < 0.5:
            suggested_exp = 1
        
        # REGRA 5: Verifica se direção final conflita muito com original
        # (log informativo, mas aceita a decisão da IA)
        if final_direction != direction:
            reason += f" | mudou de {direction}"
        
        return {
            "should_enter": should_enter,
            "direction": final_direction,
            "reason": reason,
            "suggested_exp": suggested_exp,
            "confidence": confidence,
            "trend": trend,
            "micro_trend": micro_trend
        }


# Instancia global
_fixer_instance = None

def get_ai_fixer(auto_apply: bool = False) -> AIAutoFixer:
    """Obtem instancia global do AI Fixer."""
    global _fixer_instance
    if _fixer_instance is None:
        _fixer_instance = AIAutoFixer(auto_apply=auto_apply)
    return _fixer_instance


# Funcoes de conveniencia
def ai_should_enter(context: Dict) -> Tuple[bool, str]:
    """Verifica com IA se deve entrar na operacao."""
    fixer = get_ai_fixer()
    return fixer.should_enter(context)


def ai_analyze_loss(trade_info: Dict, candles_df: pd.DataFrame) -> Dict:
    """Analisa um LOSS com IA."""
    fixer = get_ai_fixer()
    return fixer.analyze_loss(trade_info, candles_df)


def ai_record_win(trade_info: Dict, candles_df: pd.DataFrame):
    """Registra um WIN para aprendizado."""
    fixer = get_ai_fixer()
    fixer.record_win(trade_info, candles_df)


if __name__ == "__main__":
    # Teste
    print("=== TESTE AI AUTO-FIXER (APRENDIZADO) ===\n")
    
    fixer = AIAutoFixer()
    print(f"Stats: {fixer.get_stats()}\n")
    
    # Teste 1: Primeira entrada (deve PERMITIR - sem regras ainda)
    test1 = {
        "ativo": "EURUSD-OTC",
        "direction": "CALL",
        "pattern": "engolfo_alta",
        "ml_prob": 0.65,
        "trend": "DOWN",
        "hour": 10
    }
    should, reason = fixer.should_enter(test1)
    print(f"Teste 1 - CALL em DOWN (primeira vez):")
    print(f"  Permitir: {should} | Motivo: {reason}\n")
    
    # Simula 3 LOSS para criar regra
    df_fake = pd.DataFrame([
        {"open": 1.10, "high": 1.11, "low": 1.09, "close": 1.095},
        {"open": 1.095, "high": 1.10, "low": 1.08, "close": 1.085},
        {"open": 1.085, "high": 1.09, "low": 1.07, "close": 1.075},
        {"open": 1.075, "high": 1.08, "low": 1.06, "close": 1.065},
        {"open": 1.065, "high": 1.07, "low": 1.05, "close": 1.055},
    ])
    
    print("Simulando 3 LOSS com mesmo padrao...")
    for i in range(3):
        loss_info = {**test1, "pnl": -10.0}
        fixer.analyze_loss(loss_info, df_fake)
        print(f"  LOSS {i+1} registrado")
    
    print(f"\nStats apos LOSS: {fixer.get_stats()}")
    print(f"Regras: {fixer.get_rules_summary()}\n")
    
    # Teste 2: Mesma entrada apos aprendizado (deve BLOQUEAR)
    should2, reason2 = fixer.should_enter(test1)
    print(f"Teste 2 - CALL em DOWN (apos 3 losses):")
    print(f"  Permitir: {should2} | Motivo: {reason2}\n")
    
    # Teste 3: Entrada diferente (deve PERMITIR)
    test3 = {
        "ativo": "EURUSD-OTC",
        "direction": "PUT",
        "pattern": "engolfo_baixa",
        "ml_prob": 0.70,
        "trend": "DOWN",
        "hour": 14
    }
    should3, reason3 = fixer.should_enter(test3)
    print(f"Teste 3 - PUT em DOWN (padrao diferente):")
    print(f"  Permitir: {should3} | Motivo: {reason3}")
