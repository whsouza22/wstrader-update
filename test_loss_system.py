# -*- coding: utf-8 -*-
"""
Testes do Sistema de Análise de Loss
"""

import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def test_1_check_modules():
    """Teste 1: Verificar se os módulos foram criados"""
    print("\n" + "="*60)
    print("TESTE 1: Verificando módulos")
    print("="*60)
    
    try:
        import loss_analyzer
        print("✅ loss_analyzer.py importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar loss_analyzer: {e}")
        return False
    
    try:
        import auto_optimizer
        print("✅ auto_optimizer.py importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar auto_optimizer: {e}")
        return False
    
    try:
        from loss_analyzer import LossAnalyzer, get_loss_analyzer
        print("✅ LossAnalyzer pode ser instanciado")
    except ImportError as e:
        print(f"❌ Erro ao importar LossAnalyzer: {e}")
        return False
    
    try:
        from auto_optimizer import AutoOptimizer, run_auto_optimization
        print("✅ AutoOptimizer pode ser instanciado")
    except ImportError as e:
        print(f"❌ Erro ao importar AutoOptimizer: {e}")
        return False
    
    return True


def test_2_loss_analyzer_creation():
    """Teste 2: Criar instância do LossAnalyzer"""
    print("\n" + "="*60)
    print("TESTE 2: Criando LossAnalyzer")
    print("="*60)
    
    try:
        from loss_analyzer import LossAnalyzer
        
        analyzer = LossAnalyzer("http://localhost:8000")
        print(f"✅ LossAnalyzer criado: {analyzer}")
        print(f"   Backend URL: {analyzer.backend_url}")
        
        return True
    except Exception as e:
        print(f"❌ Erro ao criar LossAnalyzer: {e}")
        return False


def test_3_auto_optimizer_creation():
    """Teste 3: Criar instância do AutoOptimizer"""
    print("\n" + "="*60)
    print("TESTE 3: Criando AutoOptimizer")
    print("="*60)
    
    try:
        from auto_optimizer import AutoOptimizer
        
        optimizer = AutoOptimizer("http://localhost:8000")
        print(f"✅ AutoOptimizer criado: {optimizer}")
        print(f"   Config file: {optimizer.config_file}")
        
        # Verificar configuração padrão
        filters = optimizer.get_current_filters()
        print(f"✅ Filtros carregados: {len(filters)} parâmetros")
        
        for key, value in filters.items():
            print(f"   {key}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ Erro ao criar AutoOptimizer: {e}")
        return False


def test_4_market_analysis():
    """Teste 4: Testar análise de mercado com dados simulados"""
    print("\n" + "="*60)
    print("TESTE 4: Análise de Mercado (dados simulados)")
    print("="*60)
    
    try:
        from loss_analyzer import LossAnalyzer
        import pandas as pd
        import numpy as np
        
        analyzer = LossAnalyzer()
        
        # Criar dados simulados
        n_candles = 100
        base_price = 1.1000
        
        data = {
            'open': [],
            'close': [],
            'max': [],
            'min': [],
            'from': list(range(n_candles))
        }
        
        # Simular tendência de alta
        for i in range(n_candles):
            open_price = base_price + (i * 0.00001) + np.random.uniform(-0.00005, 0.00005)
            close_price = open_price + np.random.uniform(-0.0001, 0.0002)  # Viés de alta
            
            data['open'].append(open_price)
            data['close'].append(close_price)
            data['max'].append(max(open_price, close_price) + abs(np.random.uniform(0, 0.00005)))
            data['min'].append(min(open_price, close_price) - abs(np.random.uniform(0, 0.00005)))
        
        df = pd.DataFrame(data)
        
        # Adicionar colunas calculadas
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['max'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['min']
        df['range'] = df['max'] - df['min']
        df['is_green'] = df['close'] > df['open']
        
        # Executar análise
        market_context = analyzer.analyze_market_context(df)
        
        print("✅ Análise de contexto executada")
        print(f"   Tendência: {market_context.get('trend')}")
        print(f"   Velas verdes: {market_context.get('green_candles')}")
        print(f"   Velas vermelhas: {market_context.get('red_candles')}")
        print(f"   Volatilidade: {market_context.get('volatility')}")
        print(f"   ATR: {market_context.get('atr', 0):.5f}")
        
        return True
    except Exception as e:
        print(f"❌ Erro na análise de mercado: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_entry_analysis():
    """Teste 5: Testar análise de qualidade de entrada"""
    print("\n" + "="*60)
    print("TESTE 5: Análise de Entrada (dados simulados)")
    print("="*60)
    
    try:
        from loss_analyzer import LossAnalyzer
        import pandas as pd
        import numpy as np
        
        analyzer = LossAnalyzer()
        
        # Criar dados simulados com entrada CALL
        n_candles = 20
        base_price = 1.1000
        
        data = {
            'open': [],
            'close': [],
            'max': [],
            'min': []
        }
        
        # Últimas velas com tendência de alta (favorável para CALL)
        for i in range(n_candles):
            open_price = base_price + (i * 0.00002)
            close_price = open_price + 0.00003  # Velas verdes
            
            data['open'].append(open_price)
            data['close'].append(close_price)
            data['max'].append(close_price + 0.00001)
            data['min'].append(open_price - 0.00001)
        
        df = pd.DataFrame(data)
        df['body'] = abs(df['close'] - df['open'])
        df['range'] = df['max'] - df['min']
        df['is_green'] = df['close'] > df['open']
        
        # Analisar entrada CALL
        entry_quality = analyzer.analyze_entry_quality(df, "CALL")
        
        print("✅ Análise de entrada executada")
        print(f"   Qualidade: {entry_quality.get('entry_quality')}")
        print(f"   Body ratio: {entry_quality.get('entry_body_ratio', 0):.2f}")
        print(f"   Alinhamento: {entry_quality.get('alignment_ratio', 0):.1%}")
        print(f"   Momentum: {entry_quality.get('momentum_direction')}")
        
        return True
    except Exception as e:
        print(f"❌ Erro na análise de entrada: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_ai_analysis():
    """Teste 6: Testar geração de análise com IA"""
    print("\n" + "="*60)
    print("TESTE 6: Geração de Análise com IA")
    print("="*60)
    
    try:
        from loss_analyzer import LossAnalyzer
        
        analyzer = LossAnalyzer()
        
        # Dados simulados de um loss problemático
        market_context = {
            "trend": "bearish",
            "green_candles": 5,
            "red_candles": 15,
            "price_change_percent": -0.5,
            "atr": 0.00015,
            "volatility": "low",
            "volume_ratio": 0.8,
            "is_consolidating": False,
            "near_resistance": False,
            "near_support": False,
            "current_price": 1.10000,
            "resistance": 1.10050,
            "support": 1.09950
        }
        
        entry_quality = {
            "entry_body_ratio": 0.4,
            "entry_quality": "weak",
            "alignment_ratio": 0.2,
            "momentum_direction": "wrong",
            "prev_candles_aligned": 1,
            "prev_candles_total": 5
        }
        
        # Gerar análise (CALL em tendência de baixa - erro clássico)
        analysis = analyzer.generate_ai_analysis(
            market_context, entry_quality, 
            "EURUSD-OTC", "CALL", 10.0
        )
        
        print("✅ Análise gerada com sucesso")
        print("\n" + analysis)
        
        return True
    except Exception as e:
        print(f"❌ Erro ao gerar análise: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_config_management():
    """Teste 7: Testar gerenciamento de configuração"""
    print("\n" + "="*60)
    print("TESTE 7: Gerenciamento de Configuração")
    print("="*60)
    
    try:
        from auto_optimizer import AutoOptimizer
        import os
        
        # Criar otimizador com arquivo temporário
        test_config_file = "test_auto_config.json"
        optimizer = AutoOptimizer("http://localhost:8000", test_config_file)
        
        # Teste 1: Ajuste manual
        print("📝 Testando ajuste manual...")
        optimizer.manual_adjust("MIN_TREND_ALIGNMENT", 0.75)
        
        filters = optimizer.get_current_filters()
        assert filters["MIN_TREND_ALIGNMENT"] == 0.75, "Ajuste manual falhou"
        print("✅ Ajuste manual funcionou")
        
        # Teste 2: Blacklist
        print("📝 Testando blacklist...")
        optimizer.manual_adjust("BLACKLIST_ASSETS", ["EURUSD", "GBPUSD"])
        
        blacklist = optimizer.get_blacklist()
        assert len(blacklist) == 2, "Blacklist não foi configurada"
        print(f"✅ Blacklist configurada: {blacklist}")
        
        # Teste 3: Reset
        print("📝 Testando reset...")
        optimizer.reset_to_defaults()
        
        filters = optimizer.get_current_filters()
        assert filters["MIN_TREND_ALIGNMENT"] == 0.5, "Reset falhou"
        print("✅ Reset funcionou")
        
        # Limpar arquivo de teste
        if os.path.exists(test_config_file):
            os.remove(test_config_file)
            print("✅ Arquivo de teste removido")
        
        return True
    except Exception as e:
        print(f"❌ Erro no gerenciamento de config: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_8_integration_check():
    """Teste 8: Verificar integração com ws_auto_ai_engine"""
    print("\n" + "="*60)
    print("TESTE 8: Verificando Integração")
    print("="*60)
    
    try:
        # Verificar se o arquivo foi modificado
        with open('ws_auto_ai_engine.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verificar imports
        if 'from loss_analyzer import get_loss_analyzer' in content:
            print("✅ Import do loss_analyzer encontrado")
        else:
            print("❌ Import do loss_analyzer não encontrado")
            return False
        
        if 'LOSS_ANALYZER_ENABLED' in content:
            print("✅ LOSS_ANALYZER_ENABLED definido")
        else:
            print("❌ LOSS_ANALYZER_ENABLED não encontrado")
            return False
        
        # Verificar inicialização
        if 'self.loss_analyzer' in content:
            print("✅ self.loss_analyzer inicializado")
        else:
            print("❌ self.loss_analyzer não encontrado")
            return False
        
        # Verificar chamada na análise de loss
        if 'analyzer.analyze_loss' in content or 'loss_analyzer.analyze_loss' in content:
            print("✅ Chamada para analyze_loss encontrada")
        else:
            print("❌ Chamada para analyze_loss não encontrada")
            return False
        
        print("✅ Integração verificada com sucesso")
        return True
        
    except Exception as e:
        print(f"❌ Erro na verificação de integração: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Executa todos os testes"""
    print("\n" + "="*70)
    print("🧪 EXECUTANDO TODOS OS TESTES DO SISTEMA DE ANÁLISE DE LOSS")
    print("="*70)
    
    tests = [
        ("Verificar Módulos", test_1_check_modules),
        ("LossAnalyzer Creation", test_2_loss_analyzer_creation),
        ("AutoOptimizer Creation", test_3_auto_optimizer_creation),
        ("Análise de Mercado", test_4_market_analysis),
        ("Análise de Entrada", test_5_entry_analysis),
        ("Geração de Análise IA", test_6_ai_analysis),
        ("Gerenciamento de Config", test_7_config_management),
        ("Verificar Integração", test_8_integration_check)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Erro crítico no teste '{test_name}': {e}")
            results.append((test_name, False))
    
    # Resumo
    print("\n" + "="*70)
    print("📊 RESUMO DOS TESTES")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{status} - {test_name}")
    
    print(f"\n{'='*70}")
    print(f"Resultado: {passed}/{total} testes passaram ({passed/total*100:.1f}%)")
    print("="*70)
    
    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM! Sistema pronto para usar.")
    elif passed >= total * 0.7:
        print("\n⚠️ Maioria dos testes passou. Verifique os erros acima.")
    else:
        print("\n❌ Vários testes falharam. Revise a instalação.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_num = sys.argv[1]
        
        tests = {
            "1": test_1_check_modules,
            "2": test_2_loss_analyzer_creation,
            "3": test_3_auto_optimizer_creation,
            "4": test_4_market_analysis,
            "5": test_5_entry_analysis,
            "6": test_6_ai_analysis,
            "7": test_7_config_management,
            "8": test_8_integration_check
        }
        
        if test_num in tests:
            tests[test_num]()
        else:
            print(f"❌ Teste {test_num} não encontrado")
            print("Testes disponíveis: 1-8 ou 'all'")
    else:
        run_all_tests()
