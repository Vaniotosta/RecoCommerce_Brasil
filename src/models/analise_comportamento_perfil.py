
from pathlib import Path
import pandas as pd
import numpy as np

# =====================================================
# CONFIGURACOES
# =====================================================

RAIZ = Path(__file__).resolve().parents[2]

ARQUIVO_INTERACOES = (
    RAIZ / "data" / "processed" / "interacoes.csv"
)

ARQUIVO_RECOMENDACOES = (
    RAIZ / "data" / "processed" / "modelos" /
    "versoes" / "similaridade_20261102_090621" /
    "recomendacoes.csv"
)

PASTA_SAIDA = (
    RAIZ / "data" / "processed" / "modelos" / "analises"
)

DATA_CORTE = pd.Timestamp("2026-07-01")
K = 10

# =====================================================
# 1. CARREGAR DADOS
# =====================================================

def carregar_dados():
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

    print("Interacoes de treino:", len(treino))
    print("Interacoes de teste:", len(teste))

    return treino, teste


# =====================================================
# 2. CLASSIFICAR CLIENTES
# =====================================================

def classificar_clientes(treino):
    historico = (
        treino.groupby("customer_id")
        .agg(
            total_interacoes=("interaction_id", "count"),
            produtos_diferentes=("product_id", "nunique"),
            compras_treino=(
                "event_type",
                lambda x: (x == "purchase").sum()
            )
        )
        .reset_index()
    )

    todos = pd.read_csv(
        ARQUIVO_INTERACOES,
        usecols=["customer_id"]
    )["customer_id"].drop_duplicates()

    clientes = pd.DataFrame({"customer_id": todos})

    clientes = clientes.merge(
        historico,
        on="customer_id",
        how="left"
    )

    colunas = [
        "total_interacoes",
        "produtos_diferentes",
        "compras_treino"
    ]

    clientes[colunas] = clientes[colunas].fillna(0)

    clientes["produtos_diferentes"] = (
        clientes["produtos_diferentes"].astype(int)
    )

    clientes["perfil"] = pd.qcut(
        clientes["produtos_diferentes"].rank(
            method="first"
        ),
        q=3,
        labels=[
            "Baixo historico",
            "Historico intermediario",
            "Alto historico"
        ]
    )

    return clientes


# =====================================================
# 3. IDENTIFICAR COMPRAS RELEVANTES NO TESTE
# =====================================================

def calcular_compras_teste(treino, teste):
    compras = teste[
        teste["event_type"] == "purchase"
    ].copy()

    conhecidos = (
        treino.groupby("customer_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    compras["produto_novo"] = [
        produto not in conhecidos.get(cliente, set())
        for cliente, produto in zip(
            compras["customer_id"],
            compras["product_id"]
        )
    ]

    compras_relevantes = compras[
        compras["produto_novo"]
    ]

    resultado = (
        compras_relevantes.groupby("customer_id")
        .size()
        .reset_index(name="compras_relevantes_teste")
    )

    return resultado


# =====================================================
# 4. CALCULAR ACERTOS DO MODELO ITEM-ITEM
# =====================================================

def calcular_acertos(recomendacoes, compras_teste):
    recomendacoes = recomendacoes.sort_values(
        ["customer_id", "posicao"]
    )

    recomendacoes = recomendacoes[
        recomendacoes["posicao"] <= K
    ]

    listas = (
        recomendacoes.groupby("customer_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    acertos = []

    for cliente, produtos in compras_teste.items():
        lista = listas.get(cliente, set())

        quantidade = len(produtos.intersection(lista))

        acertos.append({
            "customer_id": cliente,
            "acertos_item_item": quantidade
        })

    return pd.DataFrame(acertos)


# =====================================================
# 5. CONSOLIDAR ANALISE
# =====================================================

def main():
    if not ARQUIVO_INTERACOES.exists():
        raise FileNotFoundError(ARQUIVO_INTERACOES)

    if not ARQUIVO_RECOMENDACOES.exists():
        raise FileNotFoundError(ARQUIVO_RECOMENDACOES)

    print("\nCarregando dados...")
    treino, teste = carregar_dados()

    print("\nClassificando clientes...")
    clientes = classificar_clientes(treino)

    print("\nCalculando compras relevantes...")
    compras_teste = calcular_compras_teste(
        treino, teste
    )

    relevancia = (
        teste[
            (teste["event_type"] == "purchase")
        ]
        .groupby("customer_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    # Filtrar apenas produtos novos no histórico
    conhecidos = (
        treino.groupby("customer_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    relevancia = {
        cliente: {
            produto for produto in produtos
            if produto not in conhecidos.get(cliente, set())
        }
        for cliente, produtos in relevancia.items()
    }

    print("\nCarregando recomendacoes...")
    recomendacoes = pd.read_csv(
        ARQUIVO_RECOMENDACOES
    )

    print("Total de recomendacoes:", len(recomendacoes))

    print("\nCalculando acertos...")
    resultado_acertos = calcular_acertos(
        recomendacoes,
        relevancia
    )

    resultado = clientes.merge(
        compras_teste,
        on="customer_id",
        how="left"
    )

    resultado = resultado.merge(
        resultado_acertos,
        on="customer_id",
        how="left"
    )

    resultado[
        ["compras_relevantes_teste", "acertos_item_item"]
    ] = resultado[
        ["compras_relevantes_teste", "acertos_item_item"]
    ].fillna(0)

    resultado["taxa_acerto"] = np.where(
        resultado["compras_relevantes_teste"] > 0,
        resultado["acertos_item_item"] /
        resultado["compras_relevantes_teste"],
        0
    )

    # =================================================
    # 6. RESUMO POR PERFIL
    # =================================================

    resumo = (
        resultado.groupby("perfil", observed=False)
        .agg(
            clientes=("customer_id", "count"),
            media_interacoes=("total_interacoes", "mean"),
            media_produtos=("produtos_diferentes", "mean"),
            media_compras_treino=("compras_treino", "mean"),
            clientes_com_compras_teste=(
                "compras_relevantes_teste",
                lambda x: (x > 0).sum()
            ),
            compras_relevantes_teste=(
                "compras_relevantes_teste", "sum"
            ),
            total_acertos=("acertos_item_item", "sum"),
            taxa_acerto_media=("taxa_acerto", "mean")
        )
        .reset_index()
    )

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    arquivo_detalhado = (
        PASTA_SAIDA / "comportamento_clientes.csv"
    )

    arquivo_resumo = (
        PASTA_SAIDA / "resumo_comportamento_perfil.csv"
    )

    resultado.to_csv(
        arquivo_detalhado,
        index=False,
        encoding="utf-8-sig"
    )

    resumo.to_csv(
        arquivo_resumo,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== RESUMO DO COMPORTAMENTO =====")
    print(resumo.to_string(index=False))

    print("\n===== ANALISE CONCLUIDA =====")
    print("Arquivos salvos em:")
    print(PASTA_SAIDA)
    print(arquivo_detalhado.name)
    print(arquivo_resumo.name)


if __name__ == "__main__":
    main()
