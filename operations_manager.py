# -*- coding: utf-8 -*-
"""
Gerenciador de Operações - WS Trader
Responsável por armazenar histórico, gerar relatórios e exportar dados
"""
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class OperationsManager:
    """Gerencia histórico de operações e geração de relatórios"""

    def __init__(self, user_email: str):
        """
        Inicializa o gerenciador de operações

        Args:
            user_email: Email do usuário para identificar os arquivos
        """
        self.user_email = user_email
        self.operations_file = self._get_operations_file_path()
        self.operations = self._load_operations()

    def _get_operations_file_path(self) -> str:
        """Retorna o caminho do arquivo de operações do usuário"""
        # Cria pasta de dados se não existir
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Sanitiza email para nome de arquivo
        safe_email = self.user_email.replace('@', '_').replace('.', '_')
        return os.path.join(data_dir, f'operations_{safe_email}.json')

    def _load_operations(self) -> List[Dict]:
        """Carrega operações do arquivo JSON"""
        try:
            if os.path.exists(self.operations_file):
                with open(self.operations_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.info(f"Arquivo de operações não encontrado, criando novo: {self.operations_file}")
                return []
        except Exception as e:
            logger.error(f"Erro ao carregar operações: {e}")
            return []

    def _save_operations(self):
        """Salva operações no arquivo JSON"""
        try:
            with open(self.operations_file, 'w', encoding='utf-8') as f:
                json.dump(self.operations, f, indent=2, ensure_ascii=False)
            logger.info(f"Operações salvas com sucesso: {len(self.operations)} registros")
        except Exception as e:
            logger.error(f"Erro ao salvar operações: {e}")

    def log_operation(self, operation: Dict[str, Any]):
        """
        Registra uma nova operação

        Args:
            operation: Dicionário com dados da operação
        """
        try:
            # Adiciona timestamp se não existir
            if 'timestamp' not in operation:
                operation['timestamp'] = datetime.now().isoformat()

            # Adiciona ID único
            operation['id'] = len(self.operations) + 1

            self.operations.append(operation)
            self._save_operations()
            logger.info(f"Operação registrada: {operation.get('id')}")
        except Exception as e:
            logger.error(f"Erro ao registrar operação: {e}")

    def update_operation(self, operation_id: int, updates: Dict[str, Any]):
        """
        Atualiza uma operação existente

        Args:
            operation_id: ID da operação
            updates: Dicionário com campos a atualizar
        """
        try:
            for op in self.operations:
                if op.get('id') == operation_id:
                    op.update(updates)
                    self._save_operations()
                    logger.info(f"Operação {operation_id} atualizada")
                    return True
            logger.warning(f"Operação {operation_id} não encontrada")
            return False
        except Exception as e:
            logger.error(f"Erro ao atualizar operação: {e}")
            return False

    def get_all_operations(self) -> List[Dict]:
        """Retorna todas as operações"""
        return self.operations

    def get_operations_by_date(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Retorna operações em um período específico

        Args:
            start_date: Data inicial (ISO format)
            end_date: Data final (ISO format)
        """
        try:
            filtered = [
                op for op in self.operations
                if start_date <= op.get('timestamp', '') <= end_date
            ]
            return filtered
        except Exception as e:
            logger.error(f"Erro ao filtrar operações por data: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calcula estatísticas das operações

        Returns:
            Dicionário com estatísticas
        """
        try:
            if not self.operations:
                return {
                    'total_operations': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0.0,
                    'total_profit': 0.0,
                    'avg_profit_per_win': 0.0,
                    'avg_loss_per_loss': 0.0
                }

            wins = [op for op in self.operations if op.get('result') == 'win']
            losses = [op for op in self.operations if op.get('result') == 'loss']

            total_profit = sum(op.get('profit', 0) for op in self.operations)
            avg_profit_per_win = sum(op.get('profit', 0) for op in wins) / len(wins) if wins else 0
            avg_loss_per_loss = sum(op.get('profit', 0) for op in losses) / len(losses) if losses else 0

            return {
                'total_operations': len(self.operations),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': (len(wins) / len(self.operations) * 100) if self.operations else 0,
                'total_profit': total_profit,
                'avg_profit_per_win': avg_profit_per_win,
                'avg_loss_per_loss': avg_loss_per_loss,
                'best_day': self._get_best_day(),
                'worst_day': self._get_worst_day()
            }
        except Exception as e:
            logger.error(f"Erro ao calcular estatísticas: {e}")
            return {}

    def _get_best_day(self) -> Dict[str, Any]:
        """Retorna informações do melhor dia"""
        try:
            daily_profits = {}
            for op in self.operations:
                date = op.get('timestamp', '')[:10]  # YYYY-MM-DD
                profit = op.get('profit', 0)
                daily_profits[date] = daily_profits.get(date, 0) + profit

            if not daily_profits:
                return {'date': None, 'profit': 0}

            best_date = max(daily_profits, key=daily_profits.get)
            return {'date': best_date, 'profit': daily_profits[best_date]}
        except Exception as e:
            logger.error(f"Erro ao calcular melhor dia: {e}")
            return {'date': None, 'profit': 0}

    def _get_worst_day(self) -> Dict[str, Any]:
        """Retorna informações do pior dia"""
        try:
            daily_profits = {}
            for op in self.operations:
                date = op.get('timestamp', '')[:10]
                profit = op.get('profit', 0)
                daily_profits[date] = daily_profits.get(date, 0) + profit

            if not daily_profits:
                return {'date': None, 'profit': 0}

            worst_date = min(daily_profits, key=daily_profits.get)
            return {'date': worst_date, 'profit': daily_profits[worst_date]}
        except Exception as e:
            logger.error(f"Erro ao calcular pior dia: {e}")
            return {'date': None, 'profit': 0}

    def generate_html_report(self) -> str:
        """
        Gera relatório completo em HTML

        Returns:
            String contendo HTML do relatório
        """
        stats = self.get_statistics()

        html = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WS Trader - Relatório de Operações</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1d29 0%, #252836 100%);
            color: #E8EAF6;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #1f2937;
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #3f4654;
            padding-bottom: 30px;
            margin-bottom: 40px;
        }}
        .header h1 {{
            font-size: 36px;
            color: #5B8DEF;
            margin-bottom: 10px;
        }}
        .header p {{
            color: #9CA3AF;
            font-size: 16px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: #2a2f3e;
            border-radius: 12px;
            padding: 24px;
            border: 1px solid #3f4654;
        }}
        .stat-card h3 {{
            color: #9CA3AF;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 4px;
        }}
        .stat-card .positive {{
            color: #10B981;
        }}
        .stat-card .negative {{
            color: #EF4444;
        }}
        .stat-card .neutral {{
            color: #5B8DEF;
        }}
        .operations-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 40px;
        }}
        .operations-table thead {{
            background: #1a1d29;
        }}
        .operations-table th {{
            padding: 16px;
            text-align: left;
            font-weight: 600;
            color: #9CA3AF;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .operations-table td {{
            padding: 16px;
            border-top: 1px solid #3f4654;
        }}
        .operations-table tbody tr:hover {{
            background: #2a2f3e;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-win {{
            background: rgba(16, 185, 129, 0.2);
            color: #10B981;
        }}
        .badge-loss {{
            background: rgba(239, 68, 68, 0.2);
            color: #EF4444;
        }}
        .badge-call {{
            background: rgba(91, 141, 239, 0.2);
            color: #5B8DEF;
        }}
        .badge-put {{
            background: rgba(245, 158, 11, 0.2);
            color: #F59E0B;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 30px;
            border-top: 1px solid #3f4654;
            color: #9CA3AF;
            font-size: 14px;
        }}
        .empty-state {{
            text-align: center;
            padding: 60px 20px;
            color: #64748B;
        }}
        .empty-state svg {{
            width: 120px;
            height: 120px;
            margin-bottom: 20px;
            opacity: 0.3;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Relatório de Operações</h1>
            <p>Gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}</p>
            <p>Usuário: {self.user_email}</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total de Operações</h3>
                <div class="value neutral">{stats.get('total_operations', 0)}</div>
            </div>
            <div class="stat-card">
                <h3>Vitórias</h3>
                <div class="value positive">{stats.get('wins', 0)}</div>
            </div>
            <div class="stat-card">
                <h3>Derrotas</h3>
                <div class="value negative">{stats.get('losses', 0)}</div>
            </div>
            <div class="stat-card">
                <h3>Win Rate</h3>
                <div class="value {'positive' if stats.get('win_rate', 0) >= 60 else 'negative'}">{stats.get('win_rate', 0):.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>Lucro Total</h3>
                <div class="value {'positive' if stats.get('total_profit', 0) >= 0 else 'negative'}">R$ {stats.get('total_profit', 0):.2f}</div>
            </div>
            <div class="stat-card">
                <h3>Lucro Médio por Win</h3>
                <div class="value positive">R$ {stats.get('avg_profit_per_win', 0):.2f}</div>
            </div>
        </div>

        <h2 style="margin-bottom: 20px; color: #E8EAF6;">Histórico de Operações</h2>
"""

        if self.operations:
            html += """
        <table class="operations-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Data/Hora</th>
                    <th>Corretora</th>
                    <th>Ativo</th>
                    <th>Direção</th>
                    <th>Valor</th>
                    <th>Resultado</th>
                    <th>Lucro</th>
                </tr>
            </thead>
            <tbody>
"""
            for op in reversed(self.operations):  # Mais recentes primeiro
                timestamp = op.get('timestamp', '')
                try:
                    dt = datetime.fromisoformat(timestamp)
                    date_str = dt.strftime('%d/%m/%Y %H:%M:%S')
                except:
                    date_str = timestamp

                result = op.get('result', 'pending')
                profit = op.get('profit', 0)
                direction = op.get('direction', 'N/A')

                html += f"""
                <tr>
                    <td>#{op.get('id', 'N/A')}</td>
                    <td>{date_str}</td>
                    <td>{op.get('broker', 'N/A')}</td>
                    <td>{op.get('asset', 'N/A')}</td>
                    <td><span class="badge badge-{'call' if direction == 'CALL' else 'put'}">{direction}</span></td>
                    <td>R$ {op.get('stake', 0):.2f}</td>
                    <td><span class="badge badge-{result}">{result.upper()}</span></td>
                    <td style="color: {'#10B981' if profit >= 0 else '#EF4444'}; font-weight: bold;">R$ {profit:.2f}</td>
                </tr>
"""

            html += """
            </tbody>
        </table>
"""
        else:
            html += """
        <div class="empty-state">
            <p>📭 Nenhuma operação registrada ainda.</p>
            <p>Execute o bot para começar a ver seus resultados aqui!</p>
        </div>
"""

        html += f"""
        <div class="footer">
            <p><strong>WS Trader</strong> - Plataforma Inteligente de Trading</p>
            <p>Gerado com IA 🤖 | © {datetime.now().year}</p>
        </div>
    </div>
</body>
</html>
"""

        return html

    def export_to_json(self, file_path: str = None) -> str:
        """
        Exporta operações para arquivo JSON

        Args:
            file_path: Caminho do arquivo (opcional)

        Returns:
            Caminho do arquivo gerado
        """
        try:
            if not file_path:
                exports_dir = os.path.join(os.path.dirname(__file__), 'exports')
                os.makedirs(exports_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_path = os.path.join(exports_dir, f'operations_export_{timestamp}.json')

            export_data = {
                'export_date': datetime.now().isoformat(),
                'user_email': self.user_email,
                'statistics': self.get_statistics(),
                'operations': self.operations
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Operações exportadas para: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Erro ao exportar para JSON: {e}")
            raise

    def clear_operations(self):
        """Limpa todas as operações (use com cuidado!)"""
        try:
            self.operations = []
            self._save_operations()
            logger.info("Todas as operações foram removidas")
        except Exception as e:
            logger.error(f"Erro ao limpar operações: {e}")
