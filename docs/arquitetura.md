# Arquitetura

O projeto separa geração, validação, preparação, análise, modelagem, rastreamento e serving.

Princípios:
- scripts Python executáveis de forma independente;
- parâmetros centralizados em `config.yaml`;
- reprodutibilidade por semente aleatória;
- separação temporal entre treino e teste;
- aplicação não treina modelos durante a navegação;
- métricas offline não são tratadas como prova de impacto comercial real.
