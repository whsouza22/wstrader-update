# -*- coding: utf-8 -*-
"""
WS Trader — Setup Firestore Licenses (COMPLETO)
✅ Conecta com service account JSON (sem input)
✅ Aceita --cred (caminho do JSON)
✅ Aceita GOOGLE_APPLICATION_CREDENTIALS (env var)
✅ Tenta também:
   - backend/credentials.json
   - Projeto_WsTrader/credentials.json (raiz do projeto)
✅ Usa Firestore database_id="(default)" (multi-db)
✅ Cria/força visibilidade: /licenses/_meta
✅ Cria/atualiza 5 licenças em /licenses
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError:
    print("❌ Instale: pip install firebase-admin")
    sys.exit(1)

COLLECTION_NAME = "licenses"
DATABASE_ID = "(default)"

LICENSE_KEYS: List[str] = [
    "7553335cd2579a717bb1a96e7503a07d"
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    p = argparse.ArgumentParser(description="Cria/atualiza licenças no Firestore (WS Trader).")
    p.add_argument("--cred", type=str, default=None, help="Caminho do JSON (service account).")
    return p.parse_args()


def _clean_path(p: str) -> str:
    return os.path.expandvars(p.strip().strip('"').strip("'"))


def resolve_cred_path(cli_path: Optional[str]) -> Optional[str]:
    """
    Ordem:
    1) --cred
    2) GOOGLE_APPLICATION_CREDENTIALS
    3) backend/credentials.json (mesma pasta do script)
    4) ../credentials.json (raiz do projeto)
    5) backend/firebase_key.json (alternativo)
    """
    # 1) argumento
    if cli_path:
        p = _clean_path(cli_path)
        if os.path.exists(p):
            return p

    # 2) env var
    envp = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if envp:
        p = _clean_path(envp)
        if os.path.exists(p):
            return p

    here = os.path.dirname(os.path.abspath(__file__))

    # 3) backend/credentials.json
    p3 = os.path.join(here, "credentials.json")
    if os.path.exists(p3):
        return p3

    # 4) raiz do projeto ../credentials.json
    p4 = os.path.abspath(os.path.join(here, "..", "credentials.json"))
    if os.path.exists(p4):
        return p4

    # 5) alternativo backend/firebase_key.json
    p5 = os.path.join(here, "firebase_key.json")
    if os.path.exists(p5):
        return p5

    return None


def read_json_info(path: str) -> Tuple[Optional[str], Optional[str]]:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j.get("project_id"), j.get("client_email")


def init_firebase(cred_path: str):
    if not os.path.exists(cred_path):
        raise FileNotFoundError(f"Credencial não existe: {cred_path}")

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(cred_path))

    # ✅ importante para multi-db
    return firestore.client(database_id=DATABASE_ID)


def ensure_collection_visible(db):
    db.collection(COLLECTION_NAME).document("_meta").set(
        {"created_at": utc_now_iso(), "note": "WS Trader licenses marker"},
        merge=True,
    )


def upsert_licenses(db):
    created = 0
    updated = 0

    for i, key in enumerate(LICENSE_KEYS, start=1):
        ref = db.collection(COLLECTION_NAME).document(key)
        snap = ref.get()

        payload: Dict = {
            "license_key": key,
            "license_type": "FREE",
            "is_used": False,
            "created_at": utc_now_iso(),
            "used_at": None,
            "used_by_email": None,
        }

        if snap.exists:
            current = snap.to_dict() or {}

            # preserva uso caso já esteja usada
            if current.get("is_used") is True:
                payload["is_used"] = True
                payload["used_at"] = current.get("used_at")
                payload["used_by_email"] = current.get("used_by_email")

            ref.set(payload, merge=True)
            updated += 1
            print(f" ⚠️ Licença {i} já existia -> validada: {key[:8]}...{key[-4:]}")
        else:
            ref.set(payload)
            created += 1
            print(f" ✅ Licença {i} criada: {key[:8]}...{key[-4:]}")

    return created, updated


def show_summary(db):
    docs = list(db.collection(COLLECTION_NAME).stream())
    ids = [d.id for d in docs if d.id != "_meta"]

    print("\n" + "=" * 70)
    print("📊 RESUMO")
    print("=" * 70)
    print(f"Database: {DATABASE_ID}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Docs (inclui _meta): {len(docs)}")
    print(f"Licenças (sem _meta): {len(ids)}")

    if ids:
        print("\nIDs:")
        for x in ids:
            print(" -", x)


def main():
    args = parse_args()
    cred_path = resolve_cred_path(args.cred)

    if not cred_path:
        print("❌ Credencial não encontrada.")
        print("➡️ Use uma destas opções:")
        print('   1) python backend/setup_firebase_licenses.py --cred "C:\\caminho\\credentials.json"')
        print("   2) Defina GOOGLE_APPLICATION_CREDENTIALS com o caminho do JSON")
        print("   3) Coloque credentials.json em backend/ ou na raiz do projeto")
        sys.exit(1)

    project_id, client_email = read_json_info(cred_path)
    print("=" * 70)
    print("🔥 FIREBASE SETUP — Criar/Gravar 'licenses'")
    print("=" * 70)
    print("project_id:", project_id)
    print("client_email:", client_email)
    print("cred:", cred_path)

    db = init_firebase(cred_path)

    print("\n🧱 Forçando visibilidade da collection...")
    ensure_collection_visible(db)
    print("✅ _meta gravado.")

    print("\n📦 Gravando licenças...")
    created, updated = upsert_licenses(db)

    print(f"\n✅ Finalizado. Criadas: {created} | Já existiam: {updated}")
    show_summary(db)

    print("\n✅ Pronto. Abra Firestore > Dados e veja a collection 'licenses'.")


if __name__ == "__main__":
    main()
