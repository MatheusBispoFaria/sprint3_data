# Limitações e melhorias planejadas

- Os indicadores de risco, sentimento e oportunidade são **estimativas**, não garantias de resultado comercial.
- O alvo usado para treinar o classificador foi derivado de sinais de regras da Sprint 3, podendo reproduzir vieses desses sinais.
- O índice de confiança por reunião não é um intervalo de confiança estatístico; os IC95% do dashboard se aplicam somente a médias de amostras com pelo menos duas observações.
- A validação fornecida utiliza um conjunto de teste controlado, que deve ser complementado por transcrições inéditas revisadas por especialistas.
- A persistência SQLite é adequada para uma aplicação de demonstração ou uso com poucos usuários. Para alta concorrência e multiusuário, planejar banco gerenciado como PostgreSQL.
- A aplicação não possui autenticação nem perfis de acesso. Não publicar dados comerciais reais em servidor público sem controle de acesso, consentimento e medidas de segurança adicionais.
- As transcrições são armazenadas integralmente, em texto claro no banco. Recomenda-se política de retenção, exclusão, criptografia em repouso e anonimização.
- No Render, o volume montado em `/var/data` é necessário para a persistência entre redeploys; sem disco persistente, os dados serão apagados quando o container for substituído.
