
import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# 1. CONFIGURAÇÃO
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

ARQUIVO_INTERACOES = BASE_DIR / "data" / "processed" / "interacoes.csv"

ARQUIVO_RANKING = (
    BASE_DIR / "data" / "processed" / "modelos"
    / "ranking_popularidade.csv"
)

PASTA_RESULTADOS = BASE_DIR / "reports" / "tabelas"
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

K = 10
DATA_CORTE = pd.Timestamp("2026-07-01")


# =====================================================
# 2. CARREGAMENTO E SEPARAÇÃO TEMPORAL
# =====================================================

dados = pd.read_csv(ARQUIVO_INTERACOES)

dados["event_timestamp"] = pd.to_datetime(
    dados["event_timestamp"]
)

treino = dados[
    dados["event_timestamp"] < DATA_CORTE
].copy()

teste = dados[
    dados["event_timestamp"] >= DATA_CORTE
].copy()

ranking = pd.read_csv(ARQUIVO_RANKING)

print("Interações de treinamento:", len(treino))
print("Interações de teste:", len(teste))


# =====================================================
# 3. HISTÓRICO E PRODUTOS JÁ CONHECIDOS
# =====================================================

historico_clientes = (
    treino.groupby("customer_id")["product_id"]
    .apply(set)
    .to_dict()
)

# Produtos já conhecidos por cada cliente
# não poderão aparecer nas recomendações.

# =====================================================
# =====================================================
# 4. CONJUNTO DE RELEVANCIA
# =====================================================

compras_teste = teste[
    teste["event_type"] == "purchase"
].copy()

compras_teste = compras_teste.drop_duplicates(
    subset=["customer_id", "product_id"]
)

relevantes = {}

for cliente, grupo in compras_teste.groupby("customer_id"):

    produtos_conhecidos = historico_clientes.get(
        cliente,
        set()
    )

    produtos_novos = (
        set(grupo["product_id"])
        - produtos_conhecidos
    )

    if produtos_novos:
        relevantes[cliente] = produtos_novos

print("Clientes com relevancia:", len(relevantes))
# =====================================================
# 5. RANKING DE RECOMENDACAO
# =====================================================

produtos_populares = ranking.sort_values(
    "pontuacao_popularidade",
    ascending=False
)["product_id"].tolist()

recomendacoes = {}

for cliente in relevantes:

    produtos_conhecidos = historico_clientes.get(
        cliente,
        set()
    )

    candidatos = [
        produto
        for produto in produtos_populares
        if produto not in produtos_conhecidos
    ]

    recomendacoes[cliente] = candidatos[:K]

print("Clientes com recomendacoes:", len(recomendacoes))


# =====================================================
# 6. CÁLCULO DAS MÉTRICAS
# =====================================================


# =====================================================
# 6. CALCULO DAS METRICAS
# =====================================================

precisions = []
recalls = []
average_precisions = []

clientes_com_recomendacoes = 0
clientes_sem_recomendacoes = 0

for cliente, produtos_relevantes in relevantes.items():

    recomendados = recomendacoes.get(cliente, [])

    if recomendados:
        clientes_com_recomendacoes += 1
    else:
        clientes_sem_recomendacoes += 1

    acertos = [
        1 if produto in produtos_relevantes else 0
        for produto in recomendados
    ]

    total_acertos = sum(acertos)

    precision = total_acertos / K

    recall = total_acertos / len(produtos_relevantes)

    precisions.append(precision)
    recalls.append(recall)

    # Average Precision @ K
    soma_precisao = 0
    acertos_encontrados = 0

    for posicao, acerto in enumerate(acertos, start=1):

        if acerto:
            acertos_encontrados += 1
            precisao_posicao = acertos_encontrados / posicao
            soma_precisao += precisao_posicao

    denominador = min(len(produtos_relevantes), K)

    average_precision = soma_precisao / denominador

    average_precisions.append(average_precision)


# =====================================================
# 7. RESULTADOS
# =====================================================

resultado = pd.DataFrame({
    "metrica": [
        "Precision@10",
        "Recall@10",
        "MAP@10"
    ],
    "valor": [
        np.mean(precisions),
        np.mean(recalls),
        np.mean(average_precisions)
    ]
})

print("\n===== AVALIACAO DO BASELINE =====")
print(resultado.to_string(index=False))

print("\nClientes avaliados:", len(precisions))
print("Clientes com recomendacoes:", clientes_com_recomendacoes)
print("Clientes sem recomendacoes:", clientes_sem_recomendacoes)

resultado.to_csv(
    PASTA_RESULTADOS / "avaliacao_popularidade.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nResultado salvo em:")
print(PASTA_RESULTADOS / "avaliacao_popularidade.csv")

# =====================================================
# 7. RESULTADOS
# =====================================================

resultado = pd.DataFrame({
    "metrica": [
        "Precision@10",
        "Recall@10",
        "MAP@10"
    ],
    "valor": [
        np.mean(precisions),
        np.mean(recalls),
        np.mean(average_precisions)
    ]
})

print("\n===== AVALIAÇÃO DO BASELINE =====")
print(resultado.to_string(index=False))

print("\nClientes avaliados:", len(precisions))

resultado.to_csv(
    PASTA_RESULTADOS / "avaliacao_popularidade.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nResultado salvo em:")
print(PASTA_RESULTADOS / "avaliacao_popularidade.csv")
