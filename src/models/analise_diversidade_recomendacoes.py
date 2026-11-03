
from pathlib import Path
import pandas as pd

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
# 1. CARREGAR DADOS
# =====================================================

def carregar_dados():
    interacoes = pd.read_csv(ARQUIVO_INTERACOES)
    produtos = pd.read_csv(ARQUIVO_PRODUTOS)
    recomendacoes = pd.read_csv(ARQUIVO_RECOMENDACOES)

    interacoes["event_timestamp"] = pd.to_datetime(
        interacoes["event_timestamp"]
    )

    treino = interacoes[
        interacoes["event_timestamp"] < DATA_CORTE
    ].copy()

    print("Interacoes de treino:", len(treino))
    print("Total de recomendacoes:", len(recomendacoes))

    return treino, produtos, recomendacoes


# =====================================================
# 2. CLASSIFICAR CLIENTES POR HISTORICO
# =====================================================

def classificar_clientes(treino):
    historico = (
        treino.groupby("customer_id")
        .agg(
            produtos_historico=("product_id", "nunique"),
            interacoes_historico=("interaction_id", "count")
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

    clientes["produtos_historico"] = (
        clientes["produtos_historico"].fillna(0)
    )

    clientes["perfil"] = pd.qcut(
        clientes["produtos_historico"].rank(
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
# 3. CALCULAR DIVERSIDADE DAS RECOMENDACOES
# =====================================================

def calcular_diversidade(recomendacoes, produtos):
    recomendacoes = recomendacoes[
        recomendacoes["posicao"] <= K
    ].copy()

    recomendacoes = recomendacoes.merge(
        produtos[["product_id", "category"]],
        on="product_id",
        how="left"
    )

    resultado = (
        recomendacoes.groupby("customer_id")
        .agg(
            total_recomendacoes=("product_id", "count"),
            categorias_recomendadas=("category", "nunique"),
            categoria_principal=(
                "category",
                lambda x: x.mode().iloc[0]
                if not x.mode().empty else "Indefinida"
            )
        )
        .reset_index()
    )

    return resultado


# =====================================================
# 4. ANALISAR CATEGORIAS MAIS RECOMENDADAS
# =====================================================

def analisar_categorias(recomendacoes, produtos):
    recomendacoes = recomendacoes[
        recomendacoes["posicao"] <= K
    ].copy()

    recomendacoes = recomendacoes.merge(
        produtos[["product_id", "category"]],
        on="product_id",
        how="left"
    )

    categorias = (
        recomendacoes.groupby("category")
        .agg(
            total_recomendacoes=("product_id", "count"),
            clientes_unicos=("customer_id", "nunique")
        )
        .reset_index()
        .sort_values(
            "total_recomendacoes",
            ascending=False
        )
    )

    categorias["participacao_percentual"] = (
        categorias["total_recomendacoes"]
        / categorias["total_recomendacoes"].sum()
        * 100
    )

    return categorias


# =====================================================
# 5. COMPARAR HISTORICO E RECOMENDACOES
# =====================================================

def comparar_perfis(clientes, diversidade):
    base = clientes.merge(
        diversidade,
        on="customer_id",
        how="left"
    )

    base["categorias_recomendadas"] = (
        base["categorias_recomendadas"].fillna(0)
    )

    resumo = (
        base.groupby("perfil", observed=False)
        .agg(
            clientes=("customer_id", "count"),
            media_produtos_historico=(
                "produtos_historico", "mean"
            ),
            media_interacoes_historico=(
                "interacoes_historico", "mean"
            ),
            media_categorias_recomendadas=(
                "categorias_recomendadas", "mean"
            ),
            mediana_categorias_recomendadas=(
                "categorias_recomendadas", "median"
            ),
            clientes_com_recomendacoes=(
                "total_recomendacoes",
                lambda x: x.notna().sum()
            )
        )
        .reset_index()
    )

    return base, resumo


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
    treino, produtos, recomendacoes = carregar_dados()

    print("\nClassificando clientes...")
    clientes = classificar_clientes(treino)

    print("\nCalculando diversidade das recomendacoes...")
    diversidade = calcular_diversidade(
        recomendacoes,
        produtos
    )

    print("\nAnalisando categorias recomendadas...")
    categorias = analisar_categorias(
        recomendacoes,
        produtos
    )

    print("\nComparando historico e recomendacoes...")
    base, resumo = comparar_perfis(
        clientes,
        diversidade
    )

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    base.to_csv(
        PASTA_SAIDA / "diversidade_recomendacoes_clientes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    resumo.to_csv(
        PASTA_SAIDA / "diversidade_recomendacoes_por_perfil.csv",
        index=False,
        encoding="utf-8-sig"
    )

    categorias.to_csv(
        PASTA_SAIDA / "categorias_mais_recomendadas.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== DIVERSIDADE DAS RECOMENDACOES POR PERFIL =====")
    print(resumo.to_string(index=False))

    print("\n===== CATEGORIAS MAIS RECOMENDADAS =====")
    print(categorias.to_string(index=False))

    print("\n===== ANALISE CONCLUIDA =====")
    print("Arquivos salvos em:")
    print(PASTA_SAIDA)


if __name__ == "__main__":
    main()
