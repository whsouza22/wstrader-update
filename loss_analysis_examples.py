# -*- coding: utf-8 -*-
"""
Exemplo de uso do sistema de análise de loss e otimização automática
"""

import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def example_1_manual_analysis():
    """Exemplo 1: Análise manual de um loss"""
    print("\n" + "="*60)
    print("EXEMPLO 1: Análise Manual de Loss")
    print("="*60)
    
    from loss_analyzer import LossAnalyzer
    from iqoptionapi.stable_api import IQ_Option
    
    # Conectar à IQ Option (substitua com suas credenciais)
    EMAIL = "seu_email@example.com"
    SENHA = "sua_senha"
    
    iq = IQ_Option(EMAIL, SENHA)
    check, reason = iq.connect()
    
    if not check:
        print(f"❌ Erro ao conectar: {reason}")
        return
    
    print("✅ Conectado à IQ Option")
    
    # Criar analisador
    analyzer = LossAnalyzer("http://localhost:8000")
    
    # Simular um loss
    order_id = 123456
    ativo = "EURUSD-OTC"
    direction = "CALL"
    stake = 10.0
    setup = {"reasons": ["pullback", "sr"]}
    
    # Executar análise
    print(f"\n🔍 Analisando loss: {ativo} | {direction} | ${stake}")
    result = analyzer.analyze_loss(iq, order_id, ativo, direction, stake, setup)
    
    if result:
        print("\n✅ Análise concluída e salva no Firebase!")
        print(f"Order ID: {result['order_id']}")
        print(f"\nProblemas identificados:")
        print(result['ai_analysis'])
    else:
        print("❌ Erro na análise")


def example_2_view_statistics():
    """Exemplo 2: Visualizar estatísticas de loss"""
    print("\n" + "="*60)
    print("EXEMPLO 2: Estatísticas de Loss")
    print("="*60)
    
    import requests
    
    try:
        # Buscar estatísticas
        response = requests.get("http://localhost:8000/api/loss/statistics", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("success"):
                stats = data["statistics"]
                
                print(f"\n📊 ESTATÍSTICAS GERAIS:")
                print(f"Total de losses: {stats['total_losses']}")
                print(f"Total perdido: ${stats['total_stake_lost']:.2f}")
                print(f"Stake médio: ${stats['avg_stake']:.2f}")
                
                print(f"\n📈 DISTRIBUIÇÃO POR DIREÇÃO:")
                for direction, count in stats['direction_distribution'].items():
                    print(f"  {direction}: {count} losses")
                
                print(f"\n🏆 TOP 5 ATIVOS COM MAIS LOSS:")
                for item in stats['top_assets_with_loss']:
                    print(f"  {item['asset']}: {item['count']} losses")
                
                print(f"\n⚠️ TOP 5 PROBLEMAS MAIS COMUNS:")
                for item in stats['top_problems']:
                    print(f"  {item['problem']}: {item['count']} vezes")
            else:
                print(f"❌ Erro: {data.get('message')}")
        else:
            print(f"❌ Erro HTTP: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Erro ao buscar estatísticas: {e}")


def example_3_get_recommendations():
    """Exemplo 3: Obter recomendações automáticas"""
    print("\n" + "="*60)
    print("EXEMPLO 3: Recomendações Automáticas")
    print("="*60)
    
    import requests
    
    try:
        response = requests.get("http://localhost:8000/api/loss/recommendations", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("success"):
                print(f"\n📋 {data['total_recommendations']} recomendações baseadas em {data['based_on_losses']} losses\n")
                
                for i, rec in enumerate(data['recommendations'], 1):
                    print(f"{i}. [{rec['priority']}] {rec['category']}")
                    print(f"   Issue: {rec['issue']}")
                    print(f"   Recomendação: {rec['recommendation']}")
                    print(f"   Config: {rec['config_suggestion']}")
                    print()
            else:
                print(f"❌ Erro: {data.get('message')}")
        else:
            print(f"❌ Erro HTTP: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Erro ao buscar recomendações: {e}")


def example_4_auto_optimize():
    """Exemplo 4: Otimização automática"""
    print("\n" + "="*60)
    print("EXEMPLO 4: Otimização Automática")
    print("="*60)
    
    from auto_optimizer import AutoOptimizer
    
    # Criar otimizador
    optimizer = AutoOptimizer("http://localhost:8000")
    
    print("\n📋 Configuração ANTES da otimização:")
    filters = optimizer.get_current_filters()
    for key, value in filters.items():
        print(f"  {key}: {value}")
    
    # Executar otimização (apenas HIGH priority)
    print("\n🔧 Executando otimização...")
    result = optimizer.auto_optimize(apply_high_priority_only=True)
    
    print(f"\n✅ {result['message']}")
    print(f"Aplicados: {result['applied']}")
    print(f"Ignorados: {result['skipped']}")
    
    if result['applied'] > 0:
        print("\n📋 Configuração DEPOIS da otimização:")
        filters = optimizer.get_current_filters()
        for key, value in filters.items():
            print(f"  {key}: {value}")
        
        blacklist = optimizer.get_blacklist()
        if blacklist:
            print(f"\n🚫 Blacklist: {', '.join(blacklist)}")


def example_5_manual_adjustments():
    """Exemplo 5: Ajustes manuais"""
    print("\n" + "="*60)
    print("EXEMPLO 5: Ajustes Manuais")
    print("="*60)
    
    from auto_optimizer import AutoOptimizer
    
    optimizer = AutoOptimizer("http://localhost:8000")
    
    # Ajustar parâmetro específico
    print("\n🔧 Ajustando MIN_TREND_ALIGNMENT para 0.7...")
    optimizer.manual_adjust("MIN_TREND_ALIGNMENT", 0.7)
    
    # Adicionar ativo à blacklist
    print("🚫 Adicionando EURUSD-OTC à blacklist...")
    optimizer.manual_adjust("BLACKLIST_ASSETS", ["EURUSD-OTC"])
    
    # Verificar mudanças
    print("\n📋 Configuração atualizada:")
    filters = optimizer.get_current_filters()
    print(f"  MIN_TREND_ALIGNMENT: {filters['MIN_TREND_ALIGNMENT']}")
    
    blacklist = optimizer.get_blacklist()
    print(f"  Blacklist: {blacklist}")


def example_6_view_history():
    """Exemplo 6: Visualizar histórico de otimizações"""
    print("\n" + "="*60)
    print("EXEMPLO 6: Histórico de Otimizações")
    print("="*60)
    
    from auto_optimizer import AutoOptimizer
    
    optimizer = AutoOptimizer("http://localhost:8000")
    history = optimizer.show_optimization_history()
    
    if not history:
        print("\n📜 Nenhuma otimização no histórico ainda")
        return
    
    print(f"\n📜 {len(history)} otimizações realizadas:\n")
    
    for i, entry in enumerate(history, 1):
        print(f"{i}. [{entry['priority']}] {entry['timestamp']}")
        print(f"   {entry['recommendation']}")
        print(f"   Config: {entry['config']}")
        print()


def example_7_list_recent_losses():
    """Exemplo 7: Listar losses recentes"""
    print("\n" + "="*60)
    print("EXEMPLO 7: Últimos Losses Analisados")
    print("="*60)
    
    import requests
    
    try:
        # Buscar últimos 5 losses
        response = requests.get("http://localhost:8000/api/loss/list?limit=5", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("success"):
                analyses = data['analyses']
                
                if not analyses:
                    print("\n📋 Nenhuma análise de loss ainda")
                    return
                
                print(f"\n📋 Últimos {len(analyses)} losses:\n")
                
                for i, analysis in enumerate(analyses, 1):
                    print(f"{i}. {analysis['asset']} - {analysis['direction']} - ${analysis['stake']:.2f}")
                    print(f"   Timestamp: {analysis['timestamp']}")
                    
                    # Mostrar problemas principais
                    ai_analysis = analysis.get('ai_analysis', '')
                    if 'PROBLEMAS IDENTIFICADOS:' in ai_analysis:
                        problems_section = ai_analysis.split('PROBLEMAS IDENTIFICADOS:')[1]
                        problems_section = problems_section.split('💡 RECOMENDAÇÕES:')[0]
                        problems = [p.strip() for p in problems_section.split('\n') if p.strip() and p.strip()[0].isdigit()]
                        
                        if problems:
                            print(f"   Problemas:")
                            for problem in problems[:2]:  # Mostrar apenas 2 primeiros
                                print(f"     - {problem[3:]}")  # Remove "1. "
                    print()
            else:
                print(f"❌ Erro: {data.get('message')}")
        else:
            print(f"❌ Erro HTTP: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Erro ao listar losses: {e}")


def example_8_complete_workflow():
    """Exemplo 8: Fluxo completo - do loss à otimização"""
    print("\n" + "="*60)
    print("EXEMPLO 8: Fluxo Completo")
    print("="*60)
    
    import requests
    
    print("\n1️⃣ Verificando se há losses analisados...")
    response = requests.get("http://localhost:8000/api/loss/statistics", timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        if data.get("success"):
            total_losses = data["statistics"]["total_losses"]
            print(f"   ✅ {total_losses} losses analisados")
            
            if total_losses < 5:
                print("   ⚠️ Poucos losses - recomendável ter pelo menos 10 para otimização")
                return
        else:
            print("   ❌ Erro ao buscar estatísticas")
            return
    
    print("\n2️⃣ Buscando recomendações...")
    response = requests.get("http://localhost:8000/api/loss/recommendations", timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        if data.get("success"):
            total_recs = data["total_recommendations"]
            print(f"   ✅ {total_recs} recomendações geradas")
            
            if total_recs == 0:
                print("   ℹ️ Nenhuma recomendação - sistema está operando bem!")
                return
        else:
            print("   ❌ Erro ao buscar recomendações")
            return
    
    print("\n3️⃣ Aplicando otimizações (HIGH priority)...")
    from auto_optimizer import AutoOptimizer
    optimizer = AutoOptimizer("http://localhost:8000")
    result = optimizer.auto_optimize(apply_high_priority_only=True)
    
    print(f"   ✅ {result['applied']} ajustes aplicados")
    
    if result['applied'] > 0:
        print("\n4️⃣ Configuração atualizada:")
        filters = optimizer.get_current_filters()
        for key, value in filters.items():
            print(f"   {key}: {value}")
        
        print("\n5️⃣ Próximos passos:")
        print("   - Reinicie o bot para aplicar os novos filtros")
        print("   - Continue operando e monitorando")
        print("   - Sistema continuará aprendendo e otimizando")
    else:
        print("\n   ℹ️ Nenhum ajuste HIGH priority necessário no momento")


# Menu principal
def main():
    """Menu principal de exemplos"""
    while True:
        print("\n" + "="*60)
        print("🔍 SISTEMA DE ANÁLISE DE LOSS - EXEMPLOS")
        print("="*60)
        print("\n1. Análise Manual de Loss")
        print("2. Ver Estatísticas")
        print("3. Obter Recomendações")
        print("4. Otimização Automática")
        print("5. Ajustes Manuais")
        print("6. Ver Histórico")
        print("7. Listar Losses Recentes")
        print("8. Fluxo Completo (Recomendado)")
        print("\n0. Sair")
        
        choice = input("\nEscolha uma opção: ").strip()
        
        if choice == "1":
            example_1_manual_analysis()
        elif choice == "2":
            example_2_view_statistics()
        elif choice == "3":
            example_3_get_recommendations()
        elif choice == "4":
            example_4_auto_optimize()
        elif choice == "5":
            example_5_manual_adjustments()
        elif choice == "6":
            example_6_view_history()
        elif choice == "7":
            example_7_list_recent_losses()
        elif choice == "8":
            example_8_complete_workflow()
        elif choice == "0":
            print("\n👋 Até logo!")
            break
        else:
            print("\n❌ Opção inválida")
        
        input("\nPressione ENTER para continuar...")


if __name__ == "__main__":
    # Você pode executar exemplos individuais ou o menu interativo
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        
        examples = {
            "1": example_1_manual_analysis,
            "2": example_2_view_statistics,
            "3": example_3_get_recommendations,
            "4": example_4_auto_optimize,
            "5": example_5_manual_adjustments,
            "6": example_6_view_history,
            "7": example_7_list_recent_losses,
            "8": example_8_complete_workflow
        }
        
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"❌ Exemplo {example_num} não encontrado")
            print("Exemplos disponíveis: 1-8")
    else:
        # Menu interativo
        main()
