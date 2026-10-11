"""Persistência local das análises em SQLite (sem dependências adicionais)."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "runtime" / "reunioes.sqlite3"


def database_path() -> Path:
    return Path(os.environ.get("DATABASE_PATH", str(DEFAULT_PATH))).expanduser().resolve()


def _connection() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_database() -> None:
    with closing(_connection()) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS reunioes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cliente TEXT NOT NULL DEFAULT '',
                    assunto TEXT NOT NULL DEFAULT '',
                    data_reuniao TEXT,
                    origem TEXT NOT NULL DEFAULT 'dashboard',
                    transcricao TEXT NOT NULL CHECK (length(trim(transcricao)) > 0),
                    classe_final TEXT NOT NULL,
                    risco_final REAL NOT NULL,
                    oportunidade REAL NOT NULL,
                    sentimento TEXT NOT NULL,
                    modelo TEXT,
                    analise_json TEXT NOT NULL,
                    criado_em TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );
                CREATE INDEX IF NOT EXISTS idx_reunioes_criado ON reunioes(criado_em DESC);
                CREATE INDEX IF NOT EXISTS idx_reunioes_risco ON reunioes(classe_final);
            """)


def _as_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["analise"] = json.loads(result.pop("analise_json"))
    return result


def salvar_reuniao(*, texto: str, analise: dict, cliente: str = "", assunto: str = "",
                  data_reuniao: str | None = None, origem: str = "dashboard") -> dict:
    texto = (texto or "").strip()
    if not texto:
        raise ValueError("Informe uma transcrição.")
    if len(texto) > 200_000:
        raise ValueError("A transcrição excede o limite de 200.000 caracteres.")
    if data_reuniao:
        try:
            date.fromisoformat(str(data_reuniao))
        except ValueError as exc:
            raise ValueError("Data deve estar no formato AAAA-MM-DD.") from exc
    cliente = str(cliente or "").strip()[:180]
    assunto = str(assunto or "").strip()[:220]
    origem = str(origem or "dashboard")[:32]
    init_database()
    with closing(_connection()) as conn:
        with conn:
            cursor = conn.execute("""
                INSERT INTO reunioes
                (cliente, assunto, data_reuniao, origem, transcricao, classe_final,
                 risco_final, oportunidade, sentimento, modelo, analise_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cliente, assunto, data_reuniao or None, origem, texto,
                analise["classe_final"], float(analise["risco_final"]),
                float(analise["oportunidade"]), analise["sentimento"], analise.get("modelo"),
                json.dumps(analise, ensure_ascii=False, allow_nan=False),
            ))
            new_id = cursor.lastrowid
    saved = obter_reuniao(new_id)
    assert saved is not None
    return saved


def obter_reuniao(meeting_id: int) -> dict | None:
    init_database()
    with closing(_connection()) as conn:
        row = conn.execute("SELECT * FROM reunioes WHERE id = ?", (int(meeting_id),)).fetchone()
        return _as_dict(row) if row else None


def listar_reunioes(*, busca: str = "", limite: int = 100, offset: int = 0,
                   classe: str | None = None, incluir_transcricao: bool = False) -> list[dict]:
    init_database()
    limite = max(1, min(500, int(limite)))
    offset = max(0, int(offset))
    clauses, values = [], []
    if busca.strip():
        clauses.append("(cliente LIKE ? OR assunto LIKE ? OR transcricao LIKE ?)")
        like = f"%{busca.strip()[:100]}%"
        values.extend([like, like, like])
    if classe in ("Risco", "Neutro"):
        clauses.append("classe_final = ?")
        values.append(classe)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    columns = "id, cliente, assunto, data_reuniao, origem, classe_final, risco_final, oportunidade, sentimento, modelo, criado_em"
    if incluir_transcricao:
        columns += ", transcricao"
    sql = f"SELECT {columns} FROM reunioes{where} ORDER BY criado_em DESC, id DESC LIMIT ? OFFSET ?"
    with closing(_connection()) as conn:
        return [dict(row) for row in conn.execute(sql, (*values, limite, offset))]


def contar_reunioes() -> int:
    init_database()
    with closing(_connection()) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM reunioes").fetchone()[0])


def excluir_reuniao(meeting_id: int) -> bool:
    init_database()
    with closing(_connection()) as conn:
        with conn:
            result = conn.execute("DELETE FROM reunioes WHERE id = ?", (int(meeting_id),))
            return result.rowcount > 0


def todas_reunioes_resumo() -> list[dict]:
    """Somente registros gravados pelo usuário, sem carregar textos grandes no dashboard."""
    init_database()
    with closing(_connection()) as conn:
        return [dict(row) for row in conn.execute("""
            SELECT id, cliente, assunto, data_reuniao, classe_final,
                   risco_final, oportunidade, sentimento, criado_em, analise_json
            FROM reunioes ORDER BY id
        """)]
