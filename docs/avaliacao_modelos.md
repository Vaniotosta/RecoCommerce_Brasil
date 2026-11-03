# Avaliação Comparativa dos Modelos de Recomendação — RecoCommerce Brasil

## 1. Objetivo

Este documento apresenta a avaliação comparativa de três estratégias de recomendação desenvolvidas no projeto RecoCommerce Brasil:

1. **Popularidade Global:** recomenda produtos com maior pontuação de interação.
2. **Similaridade Item-Item:** utiliza similaridade de cosseno entre produtos para identificar recomendações relacionadas ao comportamento dos clientes.
3. **Modelo Híbrido:** combina similaridade Item-Item com fallback de popularidade para clientes sem histórico de interações.

O objetivo é comparar a qualidade das recomendações, a cobertura de usuários e a diversidade de produtos recomendados.

## 2. Base de dados e metodologia

O experimento utiliza uma base de dados sintética de comércio eletrônico brasileiro, com 220.000 interações registradas entre clientes e produtos.

### 2.1 Separação temporal

Foi adotada uma divisão temporal para evitar que informações futuras fossem utilizadas na construção das recomendações.

- **Treinamento:** interações anteriores a 01/07/2026.
- **Teste:** interações a partir de 01/07/2026.

| Conjunto | Interações |
|---|---:|
| Treinamento | 188.671 |
| Teste | 31.329 |
| Total | 220.000 |

### 2.2 Ponderação das interações

As interações receberam pesos diferentes conforme o nível de envolvimento do cliente:

| Tipo de interação | Peso |
|---|---:|
| Visualização (view) | 1 |
| Clique (click) | 2 |
| Adição ao carrinho (cart) | 3 |
| Compra (purchase) | 5 |

Essa ponderação permite representar diferentes níveis de interesse dos clientes pelos produtos.

### 2.3 Definição de relevância

Foram considerados relevantes os produtos comprados no período de teste que não constavam no histórico de treinamento do respectivo cliente.

A avaliação contemplou:

- 906 clientes com pelo menos uma compra relevante;
- 932 produtos relevantes no conjunto de teste;
- 10 recomendações por cliente como limite de avaliação.

Todos os modelos foram avaliados utilizando o mesmo conjunto de clientes, a mesma separação temporal e os mesmos critérios de relevância.

## 3. Ambiente experimental

A implementação foi desenvolvida em Python, utilizando:

- Pandas e NumPy para manipulação de dados;
- SciPy para construção da matriz esparsa;
- Scikit-learn para cálculo da similaridade de cosseno.

A matriz de interação utilizada na recomendação possui 19.996 clientes e 2.000 produtos.

## 4. Métricas de avaliação

Foram utilizadas as seguintes métricas:

- **Precision@10:** proporção de recomendações que correspondem a produtos relevantes, considerando as dez posições.
- **Recall@10:** proporção dos produtos relevantes que foram encontrados entre as recomendações.
- **MAP@10:** média da precisão acumulada nas posições em que produtos relevantes aparecem.
- **Cobertura de clientes:** proporção dos clientes avaliados que receberam recomendações.
- **Cobertura do catálogo:** proporção dos produtos disponíveis que apareceram nas recomendações.

## 5. Resultados comparativos

Os resultados obtidos na execução experimental foram:

| Métrica | Popularidade Global | Similaridade Item-Item | Híbrido |
|---|---:|---:|---:|
| Precision@10 | 0,000221 | 0,002539 | 0,002539 |
| Recall@10 | 0,002208 | 0,025386 | 0,025386 |
| MAP@10 | 0,000552 | 0,005679 | 0,005679 |
| Cobertura de clientes | 100% | 99,89% | 100% |
| Cobertura do catálogo | 0,60% | 94,75% | 94,75% |
| Clientes avaliados | 906 | 906 | 906 |
| Clientes com recomendações | 906 | 905 | 906 |
| Produtos diferentes recomendados | 12 | 1.895 | 1.895 |

## 6. Interpretação dos resultados

### 6.1 Popularidade Global

O modelo de popularidade apresentou Precision@10 de 0,000221, Recall@10 de 0,002208 e MAP@10 de 0,000552.

Embora tenha atendido aos 906 clientes avaliados, recomendou somente 12 produtos diferentes em todo o experimento.

Esse comportamento demonstra uma concentração elevada das recomendações nos produtos mais populares do catálogo.

O modelo funciona como uma referência inicial de desempenho (*baseline*), mas apresenta limitações para personalizar as sugestões e explorar a diversidade do catálogo.

### 6.2 Similaridade Item-Item

O modelo baseado em similaridade de cosseno apresentou os melhores resultados de relevância entre as estratégias avaliadas:

- Precision@10: 0,002539;
- Recall@10: 0,025386;
- MAP@10: 0,005679.

As três métricas foram aproximadamente 11,5 vezes superiores às obtidas pelo baseline de popularidade.

Outro resultado importante foi a diversidade: o modelo recomendou 1.895 produtos diferentes, correspondendo a 94,75% dos 2.000 produtos do catálogo.

A cobertura de clientes foi de 99,89%, pois um cliente sem histórico de treinamento não recebeu recomendações pela estratégia de similaridade pura.

Os resultados indicam que a similaridade entre produtos conseguiu explorar melhor os padrões de interação presentes na base do que a recomendação baseada exclusivamente na popularidade global.

### 6.3 Modelo Híbrido

O modelo híbrido combinou a recomendação por similaridade com um mecanismo alternativo baseado em popularidade.

Seu principal resultado foi alcançar 100% de cobertura de clientes, atendendo aos 906 usuários avaliados.

As métricas de Precision@10, Recall@10 e MAP@10 permaneceram iguais às do modelo de similaridade pura.

Isso ocorreu porque o cliente que precisou utilizar o fallback não obteve acertos entre os produtos recomendados.

Portanto, o modelo híbrido resolveu a limitação de cobertura, mas não apresentou melhoria na relevância das recomendações neste experimento.

### 6.4 Análise de diversidade

A cobertura do catálogo apresentou uma diferença expressiva entre as estratégias:

- Popularidade Global: 0,60%;
- Similaridade Item-Item: 94,75%;
- Híbrido: 94,75%.

A popularidade concentrou as sugestões em poucos produtos, enquanto a similaridade distribuiu as recomendações por uma parcela muito maior do catálogo.

Essa diversidade é relevante em plataformas de comércio eletrônico, pois pode ampliar a exposição de produtos que não estão entre os mais populares.

Entretanto, maior diversidade não significa automaticamente maior qualidade ou conversão comercial. Essas hipóteses exigiriam avaliações adicionais.

## 7. Diagnóstico de Cold-Start

Foi realizado um diagnóstico individual do cliente 2360, que não possuía interações no período de treinamento.

No período de teste, esse cliente apresentou duas compras de produtos que não faziam parte de seu histórico anterior:

- Produto 1773;
- Produto 743.

O comportamento observado foi:

| Estratégia | Recomendações geradas | Acertos |
|---|---:|---:|
| Popularidade Global | 10 | 0 |
| Similaridade Item-Item | 0 | 0 |
| Híbrido | 10 | 0 |

A similaridade pura não conseguiu gerar recomendações porque não havia histórico de interações para construir o perfil do cliente.

O modelo híbrido identificou essa condição e acionou corretamente o fallback de popularidade.

Apesar de conseguir gerar dez recomendações, nenhuma correspondia aos produtos relevantes comprados pelo cliente.

O diagnóstico demonstra que o fallback resolve o problema de ausência de recomendações, mas não garante relevância para usuários novos.

## 8. Limitações identificadas

Durante a avaliação, foram identificadas as seguintes limitações:

1. **Cold-start de usuários:** clientes sem histórico não possuem informações comportamentais suficientes para recomendações personalizadas.
2. **Popularidade global:** a concentração em poucos produtos reduz a diversidade das sugestões.
3. **Relevância das recomendações:** as métricas obtidas indicam que ainda existe espaço para melhorar a correspondência entre produtos recomendados e produtos comprados.
4. **Dados sintéticos:** a base utilizada foi gerada artificialmente e não representa necessariamente os padrões de comportamento de consumidores reais.
5. **Ausência de validação comercial:** não foram realizados testes A/B nem medições reais de conversão, receita incremental ou satisfação dos usuários.

Os resultados devem ser interpretados dentro das condições desse experimento, sem generalização direta para operações comerciais reais.

## 9. Próximas melhorias

Como evolução do projeto, são propostas as seguintes melhorias:

- Desenvolver popularidade segmentada por região e categoria.
- Incorporar informações de cadastro e preferências declaradas para usuários sem histórico.
- Investigar estratégias de recomendação híbrida com maior personalização.
- Avaliar a diversidade e a novidade das recomendações com métricas específicas.
- Realizar novos experimentos com diferentes configurações e parâmetros.
- Avaliar os modelos com dados reais, quando disponíveis.

## 10. Conclusão

O experimento demonstrou que a abordagem de similaridade Item-Item apresentou desempenho superior ao baseline de popularidade nas métricas de relevância utilizadas.

O modelo híbrido ampliou a cobertura de usuários para 100%, mantendo os indicadores de relevância da similaridade pura.

A avaliação também permitiu identificar limitações importantes, especialmente o problema de cold-start e a necessidade de estratégias mais contextualizadas para novos usuários.

O desenvolvimento do RecoCommerce Brasil evidencia a aplicação de técnicas de filtragem colaborativa, engenharia de atributos comportamentais, avaliação de modelos e análise crítica de resultados.

Mais do que implementar algoritmos, o projeto buscou comparar abordagens, identificar limitações e propor melhorias orientadas ao contexto de negócio.