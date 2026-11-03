
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURACOES
# ============================================================

RAIZ = Path(__file__).resolve().parents[2]

ARQUIVO_INTERACOES = (
    RAIZ / "data" / "processed" / "interacoes.csv"
)

PASTA_VERSOES = (
    RAIZ / "data" / "processed" / "modelos" / "versoes"
)

VERSAO_ITEM_ITEM = (
    PASTA_VERSOES / "similaridade_20261102_090621"
)

VERSAO_HIBRIDO = (
    PASTA_VERSOES / "hibrido_20261102_092801"
)

PASTA_SAIDA = (
    RAIZ / "data" / "processed" / "modelos" / "analises"
)

DATA_CORTE = pd.Timestamp("2026-07-01")

K = 10


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

def carregar_dados():

    print("\nCarregando dados...")

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


# ============================================================
# 3. CLASSIFICACAO DOS CLIENTES
# ============================================================

def classificar_clientes(treino):

    print("\nClassificando clientes por historico...")

    clientes = (
        treino.groupby("customer_id")["product_id"]
        .nunique()
        .reset_index(name="produtos_historico")
    )

    # Inclui clientes sem interacoes no treinamento,
    # caso existam no arquivo original.
    todos_clientes = pd.read_csv(
        ARQUIVO_INTERACOES,
        usecols=["customer_id"],
    )["customer_id"].drop_duplicates()

    clientes = (
        pd.DataFrame({"customer_id": todos_clientes})
        .merge(
            clientes,
            on="customer_id",
            how="left",
        )
    )

    clientes["produtos_historico"] = (
        clientes["produtos_historico"]
        .fillna(0)
        .astype(int)
    )

    # Divide os clientes em tres grupos de tamanho semelhante.
    # A classificacao usa somente o historico de treinamento.
    clientes["perfil"] = pd.qcut(
        clientes["produtos_historico"].rank(
            method="first"
        ),
        q=3,
        labels=[
            "Baixo historico",
            "Historico intermediario",
            "Alto historico",
        ],
    )

    print("\nDistribuicao dos perfis:")

    print(
        clientes.groupby(
            "perfil",
            observed=False,
        )["customer_id"].count()
    )

    return clientes


# ============================================================
# 4. CONSTRUCAO DAS COMPRAS RELEVANTES
# ============================================================

def construir_relevancia(treino, teste):

    compras = teste[
        teste["event_type"] == "purchase"
    ].copy()

    conhecidos = (
        treino.groupby("customer_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    # Considera relevante a compra de um produto
    # ainda nao conhecido pelo cliente no treinamento.
    compras = compras[
        [
            produto not in conhecidos.get(cliente, set())
            for cliente, produto in zip(
                compras["customer_id"],
                compras["product_id"],
            )
        ]
    ].copy()

    relevancia = (
        compras.groupby("customer_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    print("\nClientes com compras relevantes:", len(relevancia))
    print("Compras relevantes:", len(compras))

    return relevancia


# ============================================================
# 5. AVALIACAO POR PERFIL
# ============================================================

def avaliar_por_perfil(
    recomendacoes,
    relevancia,
    clientes,
    nome_modelo,
):

    recomendacoes = recomendacoes.sort_values(
        ["customer_id", "posicao"]
    )

    listas = (
        recomendacoes.groupby("customer_id")["product_id"]
        .apply(list)
        .to_dict()
    )

    base_perfis = clientes.set_index(
        "customer_id"
    )["perfil"].to_dict()

    resultados = []

    for perfil in [
        "Baixo historico",
        "Historico intermediario",
        "Alto historico",
    ]:

        precisions = []
        recalls = []
        aps = []

        total_acertos = 0
        clientes_avaliados = 0

        for cliente, produtos_relevantes in relevancia.items():

            if base_perfis.get(cliente) != perfil:
                continue

            lista = listas.get(cliente, [])[:K]

            acertos = 0
            soma_precisao = 0.0

            for posicao, produto in enumerate(
                lista,
                start=1,
            ):

                if produto in produtos_relevantes:
                    acertos += 1
                    soma_precisao += acertos / posicao

            precisao = acertos / K

            recall = (
                acertos / len(produtos_relevantes)
            )

            ap = (
                soma_precisao
                / min(len(produtos_relevantes), K)
            )

            precisions.append(precisao)
            recalls.append(recall)
            aps.append(ap)

            total_acertos += acertos
            clientes_avaliados += 1

        resultado = {
            "modelo": nome_modelo,
            "perfil": perfil,
            "clientes_avaliados": clientes_avaliados,
            "total_acertos": total_acertos,
            "precision_10": (
                float(np.mean(precisions))
                if precisions else 0.0
            ),
            "recall_10": (
                float(np.mean(recalls))
                if recalls else 0.0
            ),
            "map_10": (
                float(np.mean(aps))
                if aps else 0.0
            ),
        }

        resultados.append(resultado)

    return pd.DataFrame(resultados)


# ============================================================
# 6. EXECUCAO DA ANALISE
# ============================================================

def main():

    if not ARQUIVO_INTERACOES.exists():
        raise FileNotFoundError(ARQUIVO_INTERACOES)

    arquivo_item = (
        VERSAO_ITEM_ITEM / "recomendacoes.csv"
    )

    arquivo_hibrido = (
        VERSAO_HIBRIDO / "recomendacoes.csv"
    )

    if not arquivo_item.exists():
        raise FileNotFoundError(arquivo_item)

    if not arquivo_hibrido.exists():
        raise FileNotFoundError(arquivo_hibrido)

    treino, teste = carregar_dados()

    clientes = classificar_clientes(treino)

    relevancia = construir_relevancia(
        treino,
        teste,
    )

    print("\nCarregando recomendacoes...")

    recomendacoes_item = pd.read_csv(
        arquivo_item
    )

    recomendacoes_hibrido = pd.read_csv(
        arquivo_hibrido
    )

    print("Recomendacoes Item-Item:", len(recomendacoes_item))
    print("Recomendacoes Hibridas:", len(recomendacoes_hibrido))

    resultado_item = avaliar_por_perfil(
        recomendacoes_item,
        relevancia,
        clientes,
        "Item-Item",
    )

    resultado_hibrido = avaliar_por_perfil(
        recomendacoes_hibrido,
        relevancia,
        clientes,
        "Hibrido 95/5",
    )

    resultado_final = pd.concat(
        [
            resultado_item,
            resultado_hibrido,
        ],
        ignore_index=True,
    )

    # Comparacao entre modelos por perfil
    comparacao = resultado_final.pivot(
        index="perfil",
        columns="modelo",
        values=[
            "precision_10",
            "recall_10",
            "map_10",
        ],
    )

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    arquivo_resultado = (
        PASTA_SAIDA / "desempenho_por_perfil.csv"
    )

    resultado_final.to_csv(
        arquivo_resultado,
        index=False,
        encoding="utf-8-sig",
    )

    arquivo_comparacao = (
        PASTA_SAIDA / "comparacao_por_perfil.csv"
    )

    comparacao.to_csv(
        arquivo_comparacao,
        encoding="utf-8-sig",
    )

    # Salva a distribuicao de historico dos clientes
    arquivo_perfis = (
        PASTA_SAIDA / "perfis_clientes.csv"
    )

    clientes.to_csv(
        arquivo_perfis,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== RESULTADOS POR PERFIL =====")

    print(
        resultado_final.to_string(
            index=False
        )
    )

    print("\n===== ANALISE CONCLUIDA =====")

    print("\nArquivos salvos em:")
    print(PASTA_SAIDA)

    print("\n1.", arquivo_resultado.name)
    print("2.", arquivo_comparacao.name)
    print("3.", arquivo_perfis.name)


if __name__ == "__main__":
    main()
