# RecoCommerce Brasil

Projeto de recomendação de produtos para uma operação de varejo digital brasileira, com foco em análise de dados, sistemas de recomendação, avaliação de modelos e práticas de MLOps.

## Objetivo

Desenvolver um pipeline reproduzível para geração, preparação e análise de dados de e-commerce, construção de modelos de recomendação, avaliação de desempenho e futura disponibilização das recomendações em uma aplicação Streamlit.

## Transparência dos dados

A primeira versão utiliza dados gerados programaticamente para simular uma operação de e-commerce. Eles permitem desenvolver e testar o pipeline, mas não representam transações observadas de uma empresa real.

Portanto, os resultados apresentados são métricas experimentais obtidas nessa base e não devem ser interpretados como impacto comercial comprovado.

## Estratégias de recomendação implementadas

Foram desenvolvidas e comparadas três estratégias:

- **Popularidade Global:** recomenda produtos com maior pontuação ponderada de interações.
- **Similaridade Item-Item:** utiliza similaridade de cosseno entre produtos com base no histórico de interações dos clientes.
- **Híbrido com fallback:** utiliza similaridade Item-Item e recorre à popularidade quando o cliente não possui histórico suficiente para gerar recomendações.

As interações recebem pesos diferentes: visualização (1), clique (2), carrinho (3) e compra (5).

## Avaliação dos modelos

A avaliação utiliza divisão temporal dos dados, com treinamento em interações anteriores a julho de 2026 e teste em interações posteriores.

Foram avaliados 906 clientes com compras relevantes de produtos ainda não observados no treinamento, utilizando as métricas Precision@10, Recall@10, MAP@10, cobertura de clientes e cobertura do catálogo.

| Modelo | Precision@10 | Recall@10 | MAP@10 | Cobertura de clientes | Cobertura do catálogo |
|---|---:|---:|---:|---:|---:|
| Popularidade Global | 0,000221 | 0,002208 | 0,000552 | 100% | 0,60% |
| Similaridade Item-Item | 0,002539 | 0,025386 | 0,005679 | 99,89% | 94,75% |
| Híbrido com fallback | 0,002539 | 0,025386 | 0,005679 | 100% | 94,75% |

### Principais observações

- A Similaridade Item-Item apresentou métricas de relevância superiores às da Popularidade Global nessa avaliação.
- A Similaridade Item-Item alcançou cobertura de 94,75% do catálogo, contra 0,60% da Popularidade Global.
- O modelo híbrido aumentou a cobertura de clientes para 100%, utilizando popularidade como alternativa para clientes sem histórico.
- O fallback resolveu a ausência de recomendações para o caso de cold start identificado, mas não gerou acertos para esse cliente no conjunto de teste.

Os resultados indicam oportunidades de melhoria na relevância das recomendações, especialmente para clientes novos.

## Estrutura do projeto

- `data/raw`: dados de entrada.
- `data/processed`: dados tratados.
- `src/data`: geração e validação dos dados.
- `src/preprocessing`: limpeza e preparação.
- `src/visualization`: análises e gráficos.
- `src/models`: modelos e avaliação.
- `app`: aplicação Streamlit.
- `docs`: documentação técnica.
- `mlruns`: rastreamento local de experimentos com MLflow.

## Início rápido (Windows)

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.data.gerar_dados
```

## Documentação técnica

A documentação detalhada da avaliação, metodologia, métricas, resultados e limitações está disponível em:

[`docs/avaliacao_modelos.md`](docs/avaliacao_modelos.md)

## Próximas melhorias

1. Investigar estratégias para aumentar a relevância das recomendações.
2. Melhorar o tratamento de cold start para clientes e produtos novos.
3. Avaliar recomendações por região e categoria.
4. Explorar métricas de diversidade e novidade.
5. Ampliar o rastreamento de experimentos e modelos com MLflow.
6. Integrar os modelos selecionados à aplicação Streamlit.

## Tecnologias

Python, Pandas, NumPy, SciPy, scikit-learn, MLflow e Streamlit.

---

**RecoCommerce Brasil** — projeto de portfólio em Ciência de Dados e Sistemas de Recomendação.
### Monitoramento do MLflow

A imagem abaixo apresenta o acompanhamento dos experimentos,
com comparação das métricas e dos resultados dos modelos.
![
    Monitoramento
](mlflow.png)

![Monitoramento de experimentos no MLflow](docs/images/mlflow-monitoramento.png)
