"""Executar com python -m unittest discover -s tests -v."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storage.database import (
    contar_reunioes, excluir_reuniao, init_database, listar_reunioes, obter_reuniao, salvar_reuniao,
)

ANALISE = {
    "classe_final": "Risco", "risco_final": 73.2, "oportunidade": 51.0,
    "sentimento": "Neutro", "modelo": "Regressão Logística",
    "termos_tfidf": [{"termo": "renovação", "peso": 0.48}],
    "produtos_totvs": ["protheus"], "concorrentes": [], "padroes": [],
    "risco_regras": 80.0, "risco_ml": 67.6, "sentimento_score": 0.0,
    "confianca_scoring": 74.2, "recomendacao": "Agendar retorno", "n_palavras": 7,
    "locutores": 2,
}

class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.previous = os.environ.get("DATABASE_PATH")
        os.environ["DATABASE_PATH"] = str(Path(self.directory.name) / "nested" / "db.sqlite3")
        init_database()

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("DATABASE_PATH", None)
        else:
            os.environ["DATABASE_PATH"] = self.previous
        self.directory.cleanup()

    def test_save_read_filter_delete_and_persistence(self):
        record = salvar_reuniao(texto="Cliente relata lentidão no Protheus.", analise=ANALISE,
                                cliente="Empresa A", assunto="Renovação", data_reuniao="2026-10-08")
        self.assertEqual(record["id"], 1)
        self.assertEqual(record["analise"]["termos_tfidf"][0]["termo"], "renovação")
        self.assertEqual(contar_reunioes(), 1)
        self.assertEqual(len(listar_reunioes(busca="Protheus", classe="Risco")), 1)
        self.assertEqual(len(listar_reunioes(classe="Neutro")), 0)
        self.assertEqual(obter_reuniao(1)["cliente"], "Empresa A")
        self.assertTrue(excluir_reuniao(1))
        self.assertEqual(contar_reunioes(), 0)

    def test_invalid_data_is_not_saved(self):
        with self.assertRaisesRegex(ValueError, "transcrição"):
            salvar_reuniao(texto="", analise=ANALISE)
        with self.assertRaisesRegex(ValueError, "Data"):
            salvar_reuniao(texto="teste", analise=ANALISE, data_reuniao="invalid")
        self.assertEqual(contar_reunioes(), 0)

    def test_user_strings_are_not_sql(self):
        salvar_reuniao(texto="Teste", analise=ANALISE, cliente="'; DROP TABLE reunioes;--")
        self.assertEqual(contar_reunioes(), 1)
        self.assertEqual(len(listar_reunioes(busca="DROP TABLE")), 1)


if __name__ == "__main__":
    unittest.main()
