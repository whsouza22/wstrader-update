"""
API SIMPLES com Firebase - Lê chaves do .env e verifica se já foram usadas
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import os
import json

# Firebase
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_ENABLED = True
except ImportError:
    FIREBASE_ENABLED = False
    print("⚠️ Firebase não instalado. Instale: pip install firebase-admin")

# ===================== APP =====================
app = FastAPI(title="WS Trader API - Firebase", version="1.0.0")

# ===================== FIREBASE SETUP =====================
db = None

@app.on_event("startup")
def init_firebase():
    """Inicializa Firebase com credenciais do ambiente ou arquivo local"""
    global db

    if not FIREBASE_ENABLED:
        print("❌ Firebase não disponível")
        return

    try:
        # Tentar carregar credenciais
        cred = None

        # Opção 1: JSON string no .env
        firebase_creds = os.getenv("FIREBASE_CREDENTIALS", "")
        if firebase_creds:
            cred_dict = json.loads(firebase_creds)
            cred = credentials.Certificate(cred_dict)
            print("[OK] Usando credenciais do .env")

        # Opção 2: Caminho definido pelo backend_server (PyInstaller)
        elif os.getenv("FIREBASE_CREDENTIALS_PATH"):
            cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                print(f"[OK] Usando {cred_path}")
            else:
                print(f"[WARN] Arquivo nao encontrado: {cred_path}")
                return

        # Opção 3: Arquivo credentials.json
        elif os.path.exists("backend/credentials.json"):
            cred = credentials.Certificate("backend/credentials.json")
            print("[OK] Usando backend/credentials.json")

        elif os.path.exists("credentials.json"):
            cred = credentials.Certificate("credentials.json")
            print("[OK] Usando credentials.json")

        else:
            print("[WARN] FIREBASE_CREDENTIALS nao configurado")
            return

        # Inicializa Firebase
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        # Usar database_id padrão (igual ao setup)
        db = firestore.client(database_id="(default)")

        print("[OK] Firebase inicializado com sucesso!")

    except Exception as e:
        print(f"[ERROR] Erro ao inicializar Firebase: {e}")

# ===================== MODELS =====================
class LicenseCheckRequest(BaseModel):
    license_key: str
    email: Optional[str] = None
    used_by_email: Optional[str] = None
    hwid: Optional[str] = None
    used_by_hwid: Optional[str] = None
    mac_address: Optional[str] = None
    mac: Optional[str] = None
    used_by_mac: Optional[str] = None
    machine_info: Optional[dict] = None

# ===================== FUNÇÕES =====================
def get_valid_keys():
    """Lê as chaves válidas das variáveis de ambiente"""
    keys = []
    for i in range(1, 6):
        key = os.getenv(f"LICENSE_KEY_{i}", "").strip().upper()
        if key:
            keys.append(key)
    return keys

# ===================== ENDPOINTS =====================
@app.get("/")
def root():
    valid_keys = get_valid_keys()
    return {
        "message": "🚀 API WS Trader Online (Firebase)",
        "version": "1.0.0",
        "total_licenses": len(valid_keys),
        "firebase_enabled": FIREBASE_ENABLED and db is not None,
        "endpoints": {
            "check_license": "POST /api/license/check",
            "admin_status": "GET /admin/status",
            "docs": "/docs"
        }
    }

@app.post("/api/license/check")
def check_license(request: LicenseCheckRequest):
    """
    Verifica se a chave existe no Firebase e se ainda não foi usada
    """
    if not FIREBASE_ENABLED or db is None:
        return {
            "valid": False,
            "message": "❌ Firebase não configurado"
        }

    license_key = request.license_key.strip().upper()
    email = (request.used_by_email or request.email or "").strip().lower()
    hwid = (request.used_by_hwid or request.hwid or "").strip()
    mac = (request.used_by_mac or request.mac_address or request.mac or "").strip().upper()

    try:
        # Buscar a chave no Firestore
        doc_ref = db.collection('licenses').document(license_key)
        doc = doc_ref.get()

        # 1. Verificar se a chave existe (ID ou campo license_key)
        if not doc.exists:
            fallback_docs = list(db.collection('licenses').where('license_key', '==', license_key).stream())
            if not fallback_docs:
                fallback_docs = list(db.collection('licenses').where('license_key', '==', license_key.lower()).stream())

            if fallback_docs:
                doc = fallback_docs[0]
                doc_ref = doc.reference
            else:
                return {
                    "valid": False,
                    "message": "❌ Chave não encontrada"
                }

        data = doc.to_dict()

        # 2. Verificar se já foi usada
        if data.get('is_used', False):
            used_email = (data.get('used_by_email') or "").strip().lower()
            used_hwid = (data.get('used_by_hwid') or "").strip()
            used_mac = (data.get('used_by_mac') or "").strip().upper()

            # Exigir Email e MAC na validação
            if not email or not mac:
                return {
                    "valid": False,
                    "message": "❌ Email e MAC são obrigatórios"
                }

            # Se o mesmo usuário + máquina (EMAIL + MAC) está validando, liberar
            email_ok = (used_email and used_email == email)
            mac_ok = (used_mac and used_mac == mac)

            # Se HWID armazenado e enviado, precisa bater também
            hwid_ok = True
            if used_hwid and hwid:
                hwid_ok = (used_hwid == hwid)

            if email_ok and mac_ok and hwid_ok:
                doc_ref.update({
                    'last_validated_at': datetime.utcnow().isoformat(),
                    'last_validated_by_email': email
                })
                return {
                    "valid": True,
                    "message": "✅ Licença válida",
                    "license_type": data.get('license_type', 'FREE'),
                    "user_data": {
                        "license_key": license_key,
                        "license_type": data.get('license_type', 'FREE')
                    }
                }

            return {
                "valid": False,
                "message": "❌ Esta chave já foi utilizada por outro email/máquina"
            }

        # 3. Marcar como usada + salvar dados da máquina
        if not email or not mac:
            return {
                "valid": False,
                "message": "❌ Email e MAC são obrigatórios"
            }


        payload = {
            'is_used': True,
            'used_at': datetime.utcnow().isoformat(),
            'used_by_email': email,
        }
        if hwid:
            payload['used_by_hwid'] = hwid
        if mac:
            payload['used_by_mac'] = mac
        if request.machine_info:
            payload['machine_info'] = request.machine_info

        doc_ref.update(payload)

        return {
            "valid": True,
            "message": "✅ Licença ativada com sucesso!",
            "license_type": data.get('license_type', 'FREE'),
            "user_data": {
                "license_key": license_key,
                "license_type": data.get('license_type', 'FREE')
            }
        }

    except Exception as e:
        return {
            "valid": False,
            "message": f"❌ Erro ao validar: {str(e)}"
        }

@app.get("/health")
def health():
    """Health check"""
    valid_keys = get_valid_keys()
    return {
        "status": "online",
        "firebase": FIREBASE_ENABLED and db is not None,
        "licenses_configured": len(valid_keys)
    }

@app.get("/admin/status")
def admin_status():
    """Ver status das licenças"""
    if not FIREBASE_ENABLED or db is None:
        return {"error": "Firebase não configurado"}

    try:
        # Buscar todas as licenças
        all_licenses = db.collection('licenses').stream()

        total = 0
        used = 0
        licenses_list = []

        for doc in all_licenses:
            # Pular documento _meta
            if doc.id == "_meta":
                continue

            total += 1
            data = doc.to_dict()

            is_used = data.get('is_used', False)
            if is_used:
                used += 1

            license_key = data.get('license_key', doc.id)
            licenses_list.append({
                "key": f"{license_key[:8]}...{license_key[-4:]}",
                "is_used": is_used,
                "used_at": data.get('used_at', None),
                "email": data.get('used_by_email', None)
            })

        return {
            "total_licenses": total,
            "used_licenses": used,
            "available_licenses": total - used,
            "licenses": licenses_list
        }

    except Exception as e:
        return {"error": f"Erro: {str(e)}"}

@app.post("/admin/reset/{license_key}")
def reset_license(license_key: str, admin_password: str):
    """Resetar uma chave (marcar como não usada)"""
    if not FIREBASE_ENABLED or db is None:
        return {"error": "Firebase não configurado"}

    # Verificar senha admin
    if admin_password != os.getenv("ADMIN_PASSWORD", ""):
        return {"error": "Senha incorreta"}

    try:
        license_key = license_key.strip().lower()
        doc_ref = db.collection('licenses').document(license_key)

        # Verificar se existe
        if not doc_ref.get().exists:
            return {"error": "Chave não encontrada"}

        # Resetar
        doc_ref.update({
            'is_used': False,
            'used_at': None,
            'used_by_email': None
        })

        return {
            "success": True,
            "message": f"✅ Chave {license_key[:8]}... resetada!"
        }
    except Exception as e:
        return {"error": f"Erro: {str(e)}"}

# ===================== LOSS ANALYSIS ENDPOINTS =====================

class LossAnalysisRequest(BaseModel):
    order_id: str
    timestamp: str
    asset: str
    direction: str
    stake: float
    market_context: dict
    entry_quality: dict
    ai_analysis: str
    setup: Optional[dict] = {}
    candles_data: Optional[dict] = {}


# ===================== WIN ANALYSIS ENDPOINTS =====================

class WinAnalysisRequest(BaseModel):
    order_id: str
    timestamp: str
    result: Optional[str] = "WIN"
    asset: str
    direction: str
    stake: float
    profit: float
    market_context: dict
    entry_quality: dict
    win_analysis: str
    setup: Optional[dict] = {}
    momentum_analysis: Optional[dict] = {}
    trend_analysis: Optional[dict] = {}
    projection_analysis: Optional[dict] = {}
    ai_prediction: Optional[dict] = {}
    chart_analysis: Optional[dict] = {}
    candles_data: Optional[dict] = {}


@app.post("/api/win/analyze")
def save_win_analysis(request: WinAnalysisRequest):
    """
    Salva uma análise de WIN no Firebase para identificar padrões vencedores
    """
    if not FIREBASE_ENABLED or db is None:
        return {
            "success": False,
            "message": "❌ Firebase não configurado"
        }

    try:
        # Prepara documento
        doc_data = {
            "order_id": request.order_id,
            "timestamp": request.timestamp,
            "result": "WIN",
            "asset": request.asset,
            "direction": request.direction,
            "stake": request.stake,
            "profit": request.profit,
            "market_context": request.market_context,
            "entry_quality": request.entry_quality,
            "win_analysis": request.win_analysis,
            "setup": request.setup or {},
            "momentum_analysis": request.momentum_analysis or {},
            "trend_analysis": request.trend_analysis or {},
            "projection_analysis": request.projection_analysis or {},
            "ai_prediction": request.ai_prediction or {},
            "chart_analysis": request.chart_analysis or {},
            "candles_data": request.candles_data or {},
            "created_at": datetime.now().isoformat()
        }

        # Salva na coleção 'win_analyses'
        doc_ref = db.collection('win_analyses').document(request.order_id)
        doc_ref.set(doc_data)

        return {
            "success": True,
            "message": f"✅ Análise de WIN salva: {request.order_id}",
            "order_id": request.order_id,
            "profit": request.profit
        }

    except Exception as e:
        print(f"[ERROR] Erro ao salvar análise de win: {e}")
        return {
            "success": False,
            "message": f"❌ Erro ao salvar: {str(e)}"
        }


@app.get("/api/win/list")
def list_win_analyses(limit: int = 50, asset: Optional[str] = None):
    """
    Lista análises de WIN do Firebase
    """
    if not FIREBASE_ENABLED or db is None:
        return {
            "success": False,
            "message": "❌ Firebase não configurado",
            "analyses": []
        }

    try:
        query = db.collection('win_analyses')
        
        # Filtrar por ativo se fornecido
        if asset:
            query = query.where('asset', '==', asset)
        
        # Ordenar por timestamp mais recente
        query = query.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
        
        docs = query.stream()
        
        analyses = []
        for doc in docs:
            data = doc.to_dict()
            analyses.append({
                "order_id": data.get("order_id"),
                "timestamp": data.get("timestamp"),
                "asset": data.get("asset"),
                "direction": data.get("direction"),
                "stake": data.get("stake"),
                "profit": data.get("profit"),
                "market_context": data.get("market_context", {}),
                "entry_quality": data.get("entry_quality", {}),
                "win_analysis": data.get("win_analysis", "")
            })

        return {
            "success": True,
            "count": len(analyses),
            "analyses": analyses
        }

    except Exception as e:
        print(f"[ERROR] Erro ao listar análises de win: {e}")
        return {
            "success": False,
            "message": f"❌ Erro ao listar: {str(e)}",
            "analyses": []
        }


@app.get("/api/win/statistics")
def get_win_statistics():
    """
    Retorna estatísticas dos WINs para análise de padrões vencedores
    """
    if not FIREBASE_ENABLED or db is None:
        return {
            "success": False,
            "message": "❌ Firebase não configurado"
        }

    try:
        docs = db.collection('win_analyses').stream()
        
        total_wins = 0
        total_profit = 0.0
        direction_wins = {"CALL": 0, "PUT": 0}
        assets_wins = {}
        best_patterns = {}
        
        for doc in docs:
            data = doc.to_dict()
            total_wins += 1
            total_profit += float(data.get("profit", 0))
            
            # Distribuição por direção
            direction = data.get("direction", "").upper()
            if direction in direction_wins:
                direction_wins[direction] += 1
            
            # Ativos com mais wins
            asset = data.get("asset", "unknown")
            assets_wins[asset] = assets_wins.get(asset, 0) + 1
            
            # Padrões vencedores
            entry_quality = data.get("entry_quality", {})
            score = entry_quality.get("score", 0)
            reasons = entry_quality.get("reasons", [])
            if reasons:
                pattern_key = "|".join(reasons[:5])  # Primeiras 5 razões
                if pattern_key not in best_patterns:
                    best_patterns[pattern_key] = {"count": 0, "total_profit": 0}
                best_patterns[pattern_key]["count"] += 1
                best_patterns[pattern_key]["total_profit"] += float(data.get("profit", 0))

        # Ordena ativos por quantidade de wins
        top_assets = sorted(assets_wins.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Ordena padrões por quantidade de wins
        top_patterns = sorted(best_patterns.items(), key=lambda x: x[1]["count"], reverse=True)[:10]

        return {
            "success": True,
            "statistics": {
                "total_wins": total_wins,
                "total_profit": round(total_profit, 2),
                "avg_profit": round(total_profit / max(1, total_wins), 2),
                "direction_distribution": direction_wins,
                "top_assets_with_wins": [{"asset": a, "count": c} for a, c in top_assets],
                "top_winning_patterns": [
                    {"pattern": p, "count": d["count"], "total_profit": round(d["total_profit"], 2)} 
                    for p, d in top_patterns
                ]
            }
        }

    except Exception as e:
        print(f"[ERROR] Erro ao obter estatísticas de win: {e}")
        return {
            "success": False,
            "message": f"❌ Erro: {str(e)}"
        }


@app.post("/api/loss/analyze")
def save_loss_analysis(request: LossAnalysisRequest):
    """
    Salva uma análise de loss no Firebase
    """
    if not FIREBASE_ENABLED or db is None:
        return {
            "success": False,
            "message": "❌ Firebase não configurado"
        }

    try:
        # Prepara documento
        doc_data = {
            "order_id": request.order_id,
            "timestamp": request.timestamp,
            "asset": request.asset,
            "direction": request.direction,
            "stake": request.stake,
            "market_context": request.market_context,
            "entry_quality": request.entry_quality,
            "ai_analysis": request.ai_analysis,
            "setup": request.setup or {},
            "candles_data": request.candles_data or {},
            "created_at": datetime.now().isoformat()
        }

        # Salva na coleção 'loss_analyses'
        doc_ref = db.collection('loss_analyses').document(request.order_id)
        doc_ref.set(doc_data)

        return {
            "success": True,
            "message": f"✅ Análise de loss salva: {request.order_id}",
            "order_id": request.order_id
        }

    except Exception as e:
        print(f"[ERROR] Erro ao salvar análise de loss: {e}")
        return {
            "success": False,
            "message": f"❌ Erro ao salvar: {str(e)}"
        }


@app.get("/api/loss/list")
def list_loss_analyses(limit: int = 50, asset: Optional[str] = None):
    """
    Lista análises de loss do Firebase
    """
    if not FIREBASE_ENABLED or db is None:
        return {
            "success": False,
            "message": "❌ Firebase não configurado",
            "analyses": []
        }

    try:
        query = db.collection('loss_analyses')
        
        # Filtrar por ativo se fornecido
        if asset:
            query = query.where('asset', '==', asset)
        
        # Ordenar por timestamp mais recente
        query = query.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
        
        docs = query.stream()
        
        analyses = []
        for doc in docs:
            data = doc.to_dict()
            analyses.append({
                "order_id": data.get("order_id"),
                "timestamp": data.get("timestamp"),
                "asset": data.get("asset"),
                "direction": data.get("direction"),
                "stake": data.get("stake"),
                "market_context": data.get("market_context", {}),
                "entry_quality": data.get("entry_quality", {}),
                "ai_analysis": data.get("ai_analysis", "")
            })

        return {
            "success": True,
            "count": len(analyses),
            "analyses": analyses
        }

    except Exception as e:
        print(f"[ERROR] Erro ao listar análises: {e}")
        return {
            "success": False,
            "message": f"❌ Erro: {str(e)}",
            "analyses": []
        }


@app.get("/api/loss/statistics")
def get_loss_statistics():
    """
    Retorna estatísticas agregadas das análises de loss
    """
    if not FIREBASE_ENABLED or db is None:
        return {
            "success": False,
            "message": "❌ Firebase não configurado"
        }

    try:
        docs = db.collection('loss_analyses').stream()
        
        total_losses = 0
        total_stake = 0.0
        problems_count = {}
        assets_count = {}
        direction_count = {"CALL": 0, "PUT": 0}
        
        for doc in docs:
            data = doc.to_dict()
            total_losses += 1
            total_stake += data.get("stake", 0.0)
            
            # Contar ativos
            asset = data.get("asset", "unknown")
            assets_count[asset] = assets_count.get(asset, 0) + 1
            
            # Contar direções
            direction = data.get("direction", "CALL")
            direction_count[direction] = direction_count.get(direction, 0) + 1
            
            # Extrair problemas comuns da análise
            analysis = data.get("ai_analysis", "")
            if "contra tendência" in analysis.lower():
                problems_count["contra_tendencia"] = problems_count.get("contra_tendencia", 0) + 1
            if "consolidação" in analysis.lower():
                problems_count["consolidacao"] = problems_count.get("consolidacao", 0) + 1
            if "resistência" in analysis.lower() or "suporte" in analysis.lower():
                problems_count["sr_forte"] = problems_count.get("sr_forte", 0) + 1
            if "entrada fraca" in analysis.lower():
                problems_count["entrada_fraca"] = problems_count.get("entrada_fraca", 0) + 1
            if "desalinhadas" in analysis.lower():
                problems_count["desalinhamento"] = problems_count.get("desalinhamento", 0) + 1
            if "alta volatilidade" in analysis.lower():
                problems_count["alta_volatilidade"] = problems_count.get("alta_volatilidade", 0) + 1

        # Top 5 ativos com mais loss
        top_assets = sorted(assets_count.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Top 5 problemas mais comuns
        top_problems = sorted(problems_count.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "success": True,
            "statistics": {
                "total_losses": total_losses,
                "total_stake_lost": round(total_stake, 2),
                "avg_stake": round(total_stake / total_losses, 2) if total_losses > 0 else 0,
                "direction_distribution": direction_count,
                "top_assets_with_loss": [{"asset": a, "count": c} for a, c in top_assets],
                "top_problems": [{"problem": p, "count": c} for p, c in top_problems]
            }
        }

    except Exception as e:
        print(f"[ERROR] Erro ao calcular estatísticas: {e}")
        return {
            "success": False,
            "message": f"❌ Erro: {str(e)}"
        }


@app.get("/api/loss/recommendations")
def get_recommendations():
    """
    Gera recomendações baseadas nas análises de loss
    """
    if not FIREBASE_ENABLED or db is None:
        return {
            "success": False,
            "message": "❌ Firebase não configurado",
            "recommendations": []
        }

    try:
        # Buscar estatísticas
        stats_response = get_loss_statistics()
        if not stats_response.get("success"):
            return stats_response
        
        stats = stats_response["statistics"]
        recommendations = []
        
        # Análise dos problemas mais comuns
        top_problems = stats.get("top_problems", [])
        
        for problem_data in top_problems:
            problem = problem_data["problem"]
            count = problem_data["count"]
            
            if problem == "contra_tendencia":
                recommendations.append({
                    "priority": "HIGH",
                    "category": "Filtro de Tendência",
                    "issue": f"{count} losses por operar contra tendência",
                    "recommendation": "Adicionar filtro: bloquear operações contra tendência quando >60% das últimas 20 velas são na direção oposta",
                    "config_suggestion": "MIN_TREND_ALIGNMENT = 0.4"
                })
            
            elif problem == "consolidacao":
                recommendations.append({
                    "priority": "HIGH",
                    "category": "Filtro de Volatilidade",
                    "issue": f"{count} losses em períodos de consolidação",
                    "recommendation": "Adicionar filtro: evitar operar quando mercado está lateral (baixa volatilidade)",
                    "config_suggestion": "MIN_VOLATILITY_RATIO = 0.8"
                })
            
            elif problem == "sr_forte":
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "Suporte/Resistência",
                    "issue": f"{count} losses próximos de S/R",
                    "recommendation": "Melhorar detecção de S/R e bloquear operações próximas (< 0.1%)",
                    "config_suggestion": "SR_MIN_DISTANCE_PERCENT = 0.1"
                })
            
            elif problem == "entrada_fraca":
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "Qualidade de Entrada",
                    "issue": f"{count} losses com vela de entrada fraca",
                    "recommendation": "Exigir corpo forte nas velas de entrada (>70% do range)",
                    "config_suggestion": "MIN_BODY_RATIO = 0.7"
                })
            
            elif problem == "desalinhamento":
                recommendations.append({
                    "priority": "HIGH",
                    "category": "Alinhamento de Velas",
                    "issue": f"{count} losses por desalinhamento",
                    "recommendation": "Exigir pelo menos 3 de 5 velas anteriores alinhadas com direção",
                    "config_suggestion": "MIN_ALIGNMENT_RATIO = 0.6"
                })
            
            elif problem == "alta_volatilidade":
                recommendations.append({
                    "priority": "LOW",
                    "category": "Gestão de Risco",
                    "issue": f"{count} losses em alta volatilidade",
                    "recommendation": "Reduzir stake em 50% quando ATR > 1.5x da média",
                    "config_suggestion": "HIGH_VOLATILITY_STAKE_REDUCTION = 0.5"
                })
        
        # Recomendações sobre ativos
        top_assets = stats.get("top_assets_with_loss", [])
        if top_assets:
            worst_asset = top_assets[0]
            if worst_asset["count"] > stats["total_losses"] * 0.3:  # Se um ativo tem >30% dos losses
                recommendations.append({
                    "priority": "HIGH",
                    "category": "Blacklist de Ativos",
                    "issue": f"{worst_asset['asset']} tem {worst_asset['count']} losses ({worst_asset['count']/stats['total_losses']*100:.1f}%)",
                    "recommendation": f"Adicionar {worst_asset['asset']} à blacklist temporária",
                    "config_suggestion": f"BLACKLIST_ASSETS = ['{worst_asset['asset']}']"
                })
        
        # Ordenar por prioridade
        priority_order = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}
        recommendations.sort(key=lambda x: priority_order[x["priority"]])

        return {
            "success": True,
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
            "based_on_losses": stats["total_losses"]
        }

    except Exception as e:
        print(f"[ERROR] Erro ao gerar recomendações: {e}")
        return {
            "success": False,
            "message": f"❌ Erro: {str(e)}",
            "recommendations": []
        }


@app.delete("/api/loss/clear")
def clear_loss_analyses():
    """
    Limpa todas as análises de loss do Firebase
    Usado para reiniciar a coleta de dados
    """
    if not FIREBASE_ENABLED or db is None:
        return {
            "success": False,
            "message": "❌ Firebase não configurado"
        }

    try:
        # Buscar todos os documentos
        docs = db.collection('loss_analyses').stream()
        
        deleted_count = 0
        for doc in docs:
            doc.reference.delete()
            deleted_count += 1
        
        print(f"[OK] {deleted_count} análises de loss deletadas")
        
        return {
            "success": True,
            "message": f"✅ {deleted_count} análises de loss deletadas com sucesso",
            "deleted_count": deleted_count
        }

    except Exception as e:
        print(f"[ERROR] Erro ao limpar análises: {e}")
        return {
            "success": False,
            "message": f"❌ Erro: {str(e)}"
        }