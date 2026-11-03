
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

    recomendacoes = recomendacoes[
        recomendacoes["posicao"] <= K
    ].copy()

    recomendacoes = recomendacoes.merge(
        produtos[["product_id", "category"]],
        on="product_id",
        how="left"
    )

    return treino, recomendacoes


# =====================================================
# 2. CLASSIFICAR CLIENTES
# =====================================================

def classificar_clientes(treino):
    historico = (
        treino.groupby("customer_id")
        .agg(
            produtos_historico=("product_id", "nunique")
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
# 3. MEDIR CONCENTRACAO POR CLIENTE
# =====================================================

def calcular_concentracao(recomendacoes):
    registros = []

    for cliente, grupo in recomendacoes.groupby("customer_id"):
        categorias = grupo["category"].dropna()

        total = len(categorias)

        if total == 0:
            continue

        contagem = categorias.value_counts()

        categorias_diferentes = contagem.size
        quantidade_categoria_dominante = contagem.max()

        participacao_dominante = (
            quantidade_categoria_dominante / total
        ) * 100

        registros.append({
            "customer_id": cliente,
            "categorias_diferentes": categorias_diferentes,
            "categoria_dominante": contagem.index[0],
            "produtos_categoria_dominante": (
                quantidade_categoria_dominante
            ),
            "participacao_categoria_dominante": (
                participacao_dominante
            )
        })

    return pd.DataFrame(registros)


# =====================================================
# 4. DISTRIBUICAO DE CATEGORIAS POR CLIENTE
# =====================================================

def calcular_distribuicao(base):
    distribuicao = (
        base.groupby(
            ["perfil", "categorias_diferentes"],
            observed=False
        )
        .agg(
            clientes=("customer_id", "count")
        )
        .reset_index()
    )

    total_por_perfil = (
        distribuicao.groupby("perfil", observed=False)["clientes"]
        .transform("sum")
    )

    distribuicao["percentual_clientes"] = (
        distribuicao["clientes"] / total_por_perfil * 100
    )

    return distribuicao


# =====================================================
# 5. RESUMO POR PERFIL
# =====================================================

def resumir_perfis(base):
    resumo = (
        base.groupby("perfil", observed=False)
        .agg(
            clientes=("customer_id", "count"),
            media_categorias_diferentes=(
                "categorias_diferentes", "mean"
            ),
            mediana_categorias_diferentes=(
                "categorias_diferentes", "median"
            ),
            media_participacao_dominante=(
                "participacao_categoria_dominante", "mean"
            ),
            mediana_participacao_dominante=(
                "participacao_categoria_dominante", "median"
            ),
            clientes_com_uma_categoria=(
                "categorias_diferentes",
                lambda x: (x == 1).sum()
            )
        )
        .reset_index()
    )

    resumo["percentual_com_uma_categoria"] = (
        resumo["clientes_com_uma_categoria"]
        / resumo["clientes"] * 100
    )

    return resumo


# =====================================================
# 6. EXECUCAO PRINCIPAL
# =====================================================

def main():
    arquivos = [
        ARQUIVO_INTERACOES,
        ARQUIVO_PRODUTOS,
        ARQUIVO_RECOMENDACOES
    ]

    for arquivo in arquivos:
        if not arquivo.exists():
            raise FileNotFoundError(
                f"Arquivo nao encontrado: {arquivo}"
            )

    print("Carregando dados...")
    treino, recomendacoes = carregar_dados()

    print("Classificando clientes...")
    clientes = classificar_clientes(treino)

    print("Calculando concentracao...")
    concentracao = calcular_concentracao(
        recomendacoes
    )

    base = clientes.merge(
        concentracao,
        on="customer_id",
        how="inner"
    )

    print("Calculando distribuicao...")
    distribuicao = calcular_distribuicao(base)

    print("Resumindo resultados...")
    resumo = resumir_perfis(base)

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    base.to_csv(
        PASTA_SAIDA / "concentracao_recomendacoes_clientes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    distribuicao.to_csv(
        PASTA_SAIDA / "distribuicao_categorias_recomendacoes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    resumo.to_csv(
        PASTA_SAIDA / "concentracao_recomendacoes_por_perfil.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== CONCENTRACAO POR PERFIL =====")
    print(resumo.to_string(index=False))

    print("\n===== DISTRIBUICAO DE CATEGORIAS =====")
    print(distribuicao.to_string(index=False))

    print("\nAnalise concluida!")
    print("Arquivos salvos em:", PASTA_SAIDA)


if __name__ == "__main__":
    main()
