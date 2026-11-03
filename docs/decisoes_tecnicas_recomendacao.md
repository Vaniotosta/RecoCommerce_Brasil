
# Decisões Técnicas — Sistema de Recomendação

## 1. Objetivo

Esta etapa prepara os dados de interação dos clientes com produtos para o desenvolvimento do sistema de recomendação do RecoCommerce Brasil.

O objetivo é transformar os eventos de comportamento em pontuações de preferência e estruturar os dados de forma eficiente para o treinamento dos modelos.

## 2. Estratégia de ponderação

Foram atribuídos pesos diferentes aos tipos de interação:

| Tipo de interação | Peso |
|---|---:|
| Visualização (view) | 1 |
| Clique (click) | 2 |
| Carrinho (cart) | 3 |
| Compra (purchase) | 5 |

A pontuação de cada combinação cliente-produto corresponde à soma dos pesos das interações registradas.

Esses pesos representam uma hipótese inicial de relevância comportamental e poderão ser ajustados durante os experimentos.

## 3. Construção da matriz cliente-produto

A partir das interações ponderadas, foi construída uma matriz com:

- 19.999 clientes;
- 2.000 produtos;
- 39.998.000 posições possíveis;
- 217.776 posições preenchidas.

## 4. Análise de esparsidade

| Indicador | Resultado |
|---|---:|
| Densidade | 0,5445% |
| Esparsidade | 99,4555% |

A elevada esparsidade demonstra que a maioria dos clientes não possui interação registrada com a maioria dos produtos.

Essa característica é comum em sistemas de recomendação baseados em comportamento implícito.

## 5. Decisão arquitetural: matriz esparsa

Foi adotada a estrutura CSR (Compressed Sparse Row), disponibilizada pela biblioteca SciPy.

A escolha reduz o armazenamento de posições vazias e permite trabalhar com estruturas maiores de maneira mais eficiente.

### Comparação de memória

| Estrutura | Memória |
|---|---:|
| Matriz densa (float32) | 152,58 MB |
| Matriz esparsa CSR | 1,74 MB |

A matriz esparsa apresentou redução aproximada de 98,86% no consumo de memória.

## 6. Arquivos produzidos

### Matrizes e identificadores

- data/processed/matrizes/matriz_cliente_produto.npz
- data/processed/matrizes/clientes_ids.npy
- data/processed/matrizes/produtos_ids.npy

### Relatórios

- reports/tabelas/relatorio_matriz_esparsa.csv

### Visualização

- reports/figures/densidade_esparsidade_matriz.png

## 7. Limitações metodológicas

- Os pesos atribuídos às interações são parâmetros iniciais, não pesos comprovadamente ótimos.
- A soma das interações não representa uma avaliação explícita de satisfação.
- Interações repetidas podem aumentar a pontuação de um produto.
- A matriz registra comportamento observado, mas não identifica necessariamente a intenção de compra.
- Os dados do projeto são gerados por programação e representam um cenário experimental, não registros operacionais de uma empresa real.

## 8. Próximas etapas

1. Avaliar a distribuição das interações por cliente e produto.
2. Definir a estratégia de treinamento e validação.
3. Desenvolver o primeiro algoritmo de recomendação.
4. Comparar os resultados com uma abordagem de referência.
5. Registrar os experimentos utilizando MLflow.
6. Integrar as recomendações ao dashboard Streamlit.

## 9. Conclusão

A preparação dos dados estabeleceu uma estrutura de interação ponderada e uma matriz esparsa eficiente para o desenvolvimento do sistema de recomendação.

A redução de memória demonstra a importância de decisões adequadas de representação dos dados antes do treinamento dos modelos.
## 2. Análise da distribuição das interações

### Distribuição por cliente

Foram identificados 19.999 clientes com interações.

- Média: 10,89 produtos diferentes por cliente.
- Mediana: 11 produtos.
- Mínimo: 1 produto.
- Máximo: 28 produtos.
- Percentil 90: 15 produtos.

Os resultados mostram que o histórico individual de produtos é relativamente limitado diante do catálogo de 2.000 produtos.

### Distribuição por produto

Foram analisados 2.000 produtos.

- Média: 108,89 clientes diferentes por produto.
- Mediana: 109 clientes.
- Mínimo: 72 clientes.
- Máximo: 155 clientes.

A quantidade de clientes por produto apresenta dispersão relativamente baixa em relação à média.

### Decisão técnica

A quantidade limitada de interações individuais reforça a necessidade de utilizar técnicas adequadas para dados esparsos.

Serão comparadas duas estratégias:

1. Recomendação baseada na popularidade dos produtos.
2. Recomendação baseada na similaridade entre produtos.

A escolha final será orientada por métricas de avaliação, e não apenas pela análise exploratória.

### Arquivos gerados

Gráficos:
- distribuicao_produtos_por_cliente.png
- distribuicao_clientes_por_produto.png
- top_20_produtos_interacoes.png

Tabelas:
- produtos_por_cliente.csv
- clientes_por_produto.csv
## 3. Estratégia de separação temporal e avaliação

### Separação dos dados

Foi adotada uma divisão temporal:

- Treinamento: interações anteriores a 01/07/2026.
- Teste: interações a partir de 01/07/2026 até 29/09/2026.

### Distribuição

Treinamento:
- 188.671 interações.
- 5.702 compras.

Teste:
- 31.329 interações.
- 954 compras.
- 928 clientes compradores.

### Análise de elegibilidade

- 927 clientes compradores possuíam histórico no treinamento.
- 932 eventos de compra envolveram produtos ainda não vistos pelo respectivo cliente.
- 906 clientes tiveram pelo menos uma compra de produto não visto.

### Decisão metodológica

A avaliação será concentrada em clientes com histórico no treinamento.

As compras de produtos ainda não vistos serão utilizadas como eventos relevantes para avaliar a capacidade de descoberta do recomendador.

Compras repetidas do mesmo produto pelo mesmo cliente serão consolidadas para evitar contagem duplicada de relevância.

O período de teste não será utilizado na construção das recomendações.

A avaliação deverá considerar métricas Precision@10, Recall@10 e MAP@10.
## 4. Primeiro modelo: recomendação por popularidade

### Objetivo

Construir um modelo de referência (baseline) que classifica os produtos pela pontuação acumulada das interações dos clientes.

### Ponderação utilizada

- Visualização: 1 ponto.
- Clique: 2 pontos.
- Carrinho: 3 pontos.
- Compra: 5 pontos.

### Resultados

- Interações utilizadas: 188.671.
- Produtos ranqueados: 2.000.
- Pontuação total: 287.293.
- Produto líder: ID 362, com 210 pontos.
- Segundo colocado: ID 990, com 208 pontos.

### Interpretação

O modelo utiliza sinais implícitos de interesse, considerando diferentes tipos de interação.

A popularidade é calculada globalmente, sem personalização por cliente.

O ranking será utilizado como baseline para comparação com modelos personalizados.

### Arquivo gerado

data/processed/modelos/ranking_popularidade.csv

### Limitação

A pontuação acumulada não representa diretamente probabilidade de compra. Produtos com muitas visualizações e cliques podem alcançar posições elevadas mesmo apresentando menos compras.
## 5. Avaliação do baseline de popularidade

### Configuração

- Corte temporal: 01/07/2026.
- Treinamento: 188.671 interações.
- Teste: 31.329 interações.
- Produtos recomendados: Top 10.
- Relevância: compra de produto ainda não conhecido pelo cliente.

### Resultados

| Métrica | Resultado |
|---|---:|
| Precision@10 | 0.000221 |
| Recall@10 | 0.002210 |
| MAP@10 | 0.000552 |
| Clientes avaliados | 905 |

### Validação dos candidatos

- Clientes com histórico: 19.996.
- Mínimo de candidatos disponíveis: 1.976.
- Clientes com menos de 10 candidatos: 0.

A disponibilidade de candidatos não explica o baixo desempenho das métricas.

### Interpretação

O baseline global de popularidade apresentou baixo desempenho na identificação das compras futuras dos clientes.

Esse resultado reforça a necessidade de investigar modelos personalizados, capazes de explorar padrões de comportamento entre clientes e produtos.

Os resultados serão utilizados como referência experimental para comparação com os próximos algoritmos.
