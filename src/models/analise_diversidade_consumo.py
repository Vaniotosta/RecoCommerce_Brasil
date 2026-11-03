
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

ARQUIVO_PRODUTOS = (
    RAIZ / "data" / "processed" / "produtos.csv"
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
# 1. CARREGAR E PREPARAR OS DADOS
# =====================================================

def carregar_dados():
    interacoes = pd.read_csv(ARQUIVO_INTERACOES)
    produtos = pd.read_csv(ARQUIVO_PRODUTOS)

    interacoes["event_timestamp"] = pd.to_datetime(
        interacoes["event_timestamp"]
    )

    dados = interacoes.merge(
        produtos[["product_id", "category"]],
        on="product_id",
        how="left"
    )

    treino = dados[
        dados["event_timestamp"] < DATA_CORTE
    ].copy()

    teste = dados[
        dados["event_timestamp"] >= DATA_CORTE
    ].copy()

    print("Interacoes de treino:", len(treino))
    print("Interacoes de teste:", len(teste))
    print("Categorias encontradas:", dados["category"].nunique())

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
            categorias_diferentes=("category", "nunique"),
            compras_treino=(
                "event_type",
                lambda x: (x == "purchase").sum()
            )
        )
        .reset_index()
    )

    todos_clientes = pd.read_csv(
        ARQUIVO_INTERACOES,
        usecols=["customer_id"]
    )["customer_id"].drop_duplicates()

    clientes = pd.DataFrame({
        "customer_id": todos_clientes
    })

    clientes = clientes.merge(
        historico,
        on="customer_id",
        how="left"
    )

    colunas = [
        "total_interacoes",
        "produtos_diferentes",
        "categorias_diferentes",
        "compras_treino"
    ]

    clientes[colunas] = clientes[colunas].fillna(0)

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
# 3. ANALISAR CATEGORIAS COMPRADAS
# =====================================================

def analisar_categorias_compradas(treino):
    compras = treino[
        treino["event_type"] == "purchase"
    ].copy()

    categorias = (
        compras.groupby("category")
        .agg(
            total_compras=("interaction_id", "count"),
            clientes_unicos=("customer_id", "nunique")
        )
        .reset_index()
        .sort_values(
            "total_compras",
            ascending=False
        )
    )

    return categorias


# =====================================================
# 4. CATEGORIAS POR PERFIL
# =====================================================

def categorias_por_perfil(treino, clientes):
    compras = treino[
        treino["event_type"] == "purchase"
    ].copy()

    compras = compras.merge(
        clientes[["customer_id", "perfil"]],
        on="customer_id",
        how="left"
    )

    resultado = (
        compras.groupby(
            ["perfil", "category"],
            observed=False
        )
        .agg(
            total_compras=("interaction_id", "count"),
            clientes_unicos=("customer_id", "nunique")
        )
        .reset_index()
        .sort_values(
            ["perfil", "total_compras"],
            ascending=[True, False]
        )
    )

    return resultado


# =====================================================
# 5. RESUMO DA DIVERSIDADE
# =====================================================

def resumo_diversidade(clientes):
    resumo = (
        clientes.groupby("perfil", observed=False)
        .agg(
            clientes=("customer_id", "count"),
            media_interacoes=("total_interacoes", "mean"),
            media_produtos=("produtos_diferentes", "mean"),
            media_categorias=("categorias_diferentes", "mean"),
            mediana_categorias=("categorias_diferentes", "median"),
            maximo_categorias=("categorias_diferentes", "max"),
            media_compras=("compras_treino", "mean")
        )
        .reset_index()
    )

    return resumo


# =====================================================
# 6. EXECUCAO PRINCIPAL
# =====================================================

def main():
    for arquivo in [
        ARQUIVO_INTERACOES,
        ARQUIVO_PRODUTOS,
        ARQUIVO_RECOMENDACOES
    ]:
        if not arquivo.exists():
            raise FileNotFoundError(arquivo)

    print("\nCarregando dados...")
    treino, teste = carregar_dados()

    print("\nClassificando clientes...")
    clientes = classificar_clientes(treino)

    print("\nCalculando diversidade...")
    resumo = resumo_diversidade(clientes)

    print("\nAnalisando categorias compradas...")
    categorias = analisar_categorias_compradas(treino)

    print("\nAnalisando categorias por perfil...")
    por_perfil = categorias_por_perfil(
        treino,
        clientes
    )

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    resumo.to_csv(
        PASTA_SAIDA / "diversidade_por_perfil.csv",
        index=False,
        encoding="utf-8-sig"
    )

    categorias.to_csv(
        PASTA_SAIDA / "categorias_mais_compradas.csv",
        index=False,
        encoding="utf-8-sig"
    )

    por_perfil.to_csv(
        PASTA_SAIDA / "categorias_por_perfil.csv",
        index=False,
        encoding="utf-8-sig"
    )

    clientes.to_csv(
        PASTA_SAIDA / "diversidade_clientes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== DIVERSIDADE POR PERFIL =====")
    print(resumo.to_string(index=False))

    print("\n===== CATEGORIAS MAIS COMPRADAS =====")
    print(categorias.to_string(index=False))

    print("\n===== CATEGORIAS POR PERFIL =====")
    for perfil in [
        "Baixo historico",
        "Historico intermediario",
        "Alto historico"
    ]:
        print(f"\n--- {perfil} ---")
        parte = por_perfil[
            por_perfil["perfil"] == perfil
        ].head(10)

        print(parte.to_string(index=False))

    print("\n===== ANALISE CONCLUIDA =====")
    print("Arquivos salvos em:")
    print(PASTA_SAIDA)


if __name__ == "__main__":
    main()
