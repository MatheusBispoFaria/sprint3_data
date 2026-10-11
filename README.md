# TOTVS Meeting Intelligence — Sprint 4

Aplicação Dash com análise de novas transcrições, armazenamento das reuniões em **SQLite** e dashboards operacionais. Usa a base original da Sprint 3 para treinar o modelo, e aproveita o motor de extração do projeto `totvs_datascience`.

## Iniciar (Windows)

Extraia o projeto, abra o terminal **na pasta que contém `app.py`** e execute:

```powershell
py -m pip install -r requirements.txt
py app.py
```

Acesse http://127.0.0.1:8050.

## Uso

1. **Transcrição:** informe cliente/assunto (opcionais), cole a reunião e clique em **Analisar e salvar**. O sistema salva o texto e a análise completa automaticamente.
2. **Dashboard:** veja gráficos e indicadores das **reuniões salvas no banco**, com filtros e IC95% nas médias quando há amostra suficiente.
3. **Reuniões salvas:** procure uma reunião, abra o registro, consulte indicadores e transcrição original, ou exclua o registro.
4. **Base de referência:** consulte a base anonimizada da Sprint 3, usada para treinamento. Ela não é o mesmo conjunto das reuniões salvas pelo usuário.

A navegação só abre ao clicar no ícone ☰. As descrições da Sprint foram transferidas para `docs/`, não são exibidas ao usuário final.

## Estrutura

```text
app.py                        # Inicialização Dash + API Flask
pages/
  home.py                     # Transcrição e resultado
  dashboard.py                # Indicadores das reuniões persistidas
  meetings.py                 # Histórico e consulta/exclusão
  overview.py                 # Base histórica de referência
analysis/
  core.py                     # Extração por regras (totvs_datascience)
  ml_pipeline.py              # TF-IDF + modelos da Sprint 3
  sprint4_analysis.py         # Scoring, sentimento e recomendações
services/
  data_service.py             # Treinamento/consulta do modelo
  reunioes.py                 # Orquestra análise + gravação
storage/
  database.py                 # Tabelas e operações SQLite
ui/
  components.py               # Componentes de interface
  analysis_view.py            # Resultado da análise
  viz.py                      # Aparência dos gráficos
assets/style.css              # Layout widescreen + sidebar sobreposta
data/reference/               # Arquivos originais, não editáveis (treinamento)
runtime/reunioes.sqlite3      # Banco criado na primeira inicialização
archive/                      # Cópia do projeto antigo para referência
docs/                         # Documentação de entrega e limitações
tests/                        # Testes automatizados do armazenamento
requirements.txt
Procfile
render.yaml
```

## Banco de dados

Banco: SQLite (já incluso no Python, sem instalação externa). A tabela `reunioes` contém cliente, assunto, data, origem, transcrição, classificação, risco, oportunidade, sentimento, nome do modelo, análise JSON e data de criação. Índices auxiliam pesquisas por data e classificação.

Por padrão, fica em `runtime/reunioes.sqlite3`. Se quiser armazenar em outro local, configure a variável de ambiente `DATABASE_PATH` com o caminho desejado. O banco permanece ao fechar e abrir o programa; **não delete a pasta `runtime/` se quiser preservar as reuniões**. O ZIP da entrega não inclui um banco de testes previamente preenchido.

A base original (`data/reference/`) permanece separada para treinamento/validação. Reuniões cadastradas **não retreinam** automaticamente o modelo — é necessário um processo de rotulagem e reavaliação para isso.

## API

| Método | Endpoint | Ação |
|---|---|---|
| GET | `/health` | Status e número de registros salvos |
| POST | `/api/analisar` | Analisa e **salva** nova transcrição (`salvar:false` apenas analisa) |
| GET/POST | `/api/reunioes` | Lista reuniões / analisa e salva |
| GET | `/api/reunioes/{id}` | Registro com transcrição e análise |
| DELETE | `/api/reunioes/{id}` | Exclui a reunião |
| GET | `/api/modelo` | Resumo das métricas de ML |
| GET | `/api/validacao` | Validação controlada |

Exemplo de JSON:

```json
{"texto":"Estamos avaliando novas integrações ao Protheus.","cliente":"Cliente exemplo","assunto":"Acompanhamento","data_reuniao":"2026-10-08"}
```

## Deploy

`gunicorn app:server` executa o servidor Flask/Dash em Linux. `render.yaml` configura um disco persistente montado em `/var/data` e `DATABASE_PATH=/var/data/reunioes.sqlite3`. **Discos persistentes podem exigir um plano pago do provedor.** Em outras plataformas, configure um volume persistente equivalente; sem ele, o SQLite pode ser perdido em atualizações/reinicializações do serviço.

**Atenção:** este protótipo não possui login. Não exponha transcrições reais confidenciais na internet antes de implementar autenticação, autorização, anonimização e proteção dos dados. Consulte `docs/LIMITACOES.md`.

## Correção de layout: gráficos com altura fixa

Os gráficos de análise e dos dashboards utilizam o componente `fixed_graph()` de `ui/viz.py`, com altura explícita, `autosize=False` e `responsive=False`. Isso elimina a realimentação de tamanho vertical que pode ocorrer entre o `dcc.Graph` (Plotly) e os painéis em grade CSS, causando crescimento contínuo da página depois de uma análise. Não altera modelos, banco nem indicadores.

## Testes

```powershell
py -m unittest discover -s tests -v
```

Para validar a interface visualmente, rode `py app.py` e teste o menu, o cadastro de reunião e o histórico em um navegador.
