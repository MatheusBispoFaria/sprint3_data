# Sprint 4 — Requisitos de Data Science

Documento técnico de avaliação. **Não aparece na interface.**

A rubrica considerada é a do PDF `SPRINT_3_E_4_CHALLENGE_2026_TOTVS` (páginas 23–24).

| Exigência | Pontos | Implementação |
| --- | ---: | --- |
| Pipeline de processamento | 15 | `analysis/core.py`, `analysis/ml_pipeline.py`, `analysis/sprint4_analysis.py`, `services/data_service.py` |
| Modelo final e justificativa | 15 | `analysis/ml_pipeline.py` (comparação de Regressão Logística e Random Forest); `GET /api/modelo` |
| Scoring/indicadores | 20 | `analysis/sprint4_analysis.py`, visualização em `ui/analysis_view.py` |
| Protótipo funcional | 20 | `pages/home.py` e `POST /api/analisar` |
| Validação com novas transcrições | 15 | `data/reference/validation_sprint4.csv`, `GET /api/validacao` |
| Recomendações de negócio | 10 | `analysis/sprint4_analysis.py`, painel de resultados |
| Limitações e próximos passos | 5 | `docs/LIMITACOES.md` |

O fluxo **Transcrição → Processamento → Modelo → Indicadores → Recomendação** permanece funcional. A entrada de texto é gravada em SQLite com a análise completa, para consulta histórica posterior. A avaliação do modelo usa rótulos derivados por regras na Sprint 3; não equivale a validação clínica/comercial externa.
