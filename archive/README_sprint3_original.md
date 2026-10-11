# Classificação de Risco de Churn — App Streamlit (Sprint 3)

App interativo da Sprint 3 de Data Science and Statistical Computing: classifica reuniões
comerciais em **Risco de Churn / Neutro** usando Machine Learning supervisionado sobre as
transcrições reais (`ANON_transcricao.json`), com rótulos derivados do dicionário de termos
(`dicionario_termos.json`).

## Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run app_ml_sprint3.py
```

Abra o link que aparecer no terminal (geralmente http://localhost:8501).

## Como publicar (Streamlit Community Cloud, gratuito)

1. Crie um repositório no GitHub e suba estes 4 arquivos (`app_ml_sprint3.py`,
   `dicionario_termos.json`, `ANON_transcricao.json`, `requirements.txt`).
2. Acesse https://share.streamlit.io, faça login com o GitHub e clique em "New app".
3. Selecione o repositório, o branch e o arquivo principal (`app_ml_sprint3.py`).
4. Clique em "Deploy" — em alguns minutos você recebe um link público para entregar.

## O que o app mostra

- **Problema & Alvo**: como o idx_risco é calculado e como o limiar define o rótulo (ajustável na barra lateral).
- **Classes & Balanceamento**: distribuição das classes e a estratégia adotada.
- **Modelos**: Regressão Logística vs Random Forest sobre TF-IDF.
- **Avaliação**: matriz de confusão, precision/recall/F1/accuracy, curva ROC, comparação entre modelos.
- **Análise de Erros**: estatísticas e inspeção individual de Falsos Negativos/Positivos.
- **Conclusões**: implicações para o negócio e conexão com o dashboard `dash.py`.
