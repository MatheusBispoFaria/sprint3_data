# Correção — crescimento contínuo da página após analisar transcrição

## Sintoma

Após clicar em **Analisar e salvar**, a área dos gráficos pode crescer repetidamente em altura, fazendo a página rolar indefinidamente.

## Ajustes

- Novo helper `ui/viz.py:fixed_graph` define altura explícita e desativa autosize/resize automático do Plotly (`responsive=False`).
- Aplicado aos gráficos do resultado, dashboard e base de referência.
- CSS limita crescimento interno dos gráficos e evita que cartões em grade expandam sua própria altura.
- Sem alterações na análise, nos indicadores, na API ou no banco de reuniões.

## Como atualizar com dados existentes

1. Feche a aplicação.
2. Faça cópia de `runtime/reunioes.sqlite3` se você já salvou reuniões.
3. Substitua os arquivos do código pelo ZIP atualizado, **sem apagar** seu banco.
4. Reinicie usando `py app.py` e atualize a página com `Ctrl+F5`.

## Verificações

- Todos os módulos compilam sem erro de sintaxe.
- Três testes do SQLite aprovados.
- Layout do helper exercitado com stub em seis alturas fixas.
- A interface visual deve ser confirmada localmente: Dash indisponível neste ambiente.
