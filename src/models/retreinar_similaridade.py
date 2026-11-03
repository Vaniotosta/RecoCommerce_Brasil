from pathlib import Path
from datetime import datetime
import json

import mlflow
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. CONFIGURACOES
# ============================================================

RAIZ = Path(__file__).resolve().parents[2]

ARQUIVO_INTERACOES = (
    RAIZ / "data" / "processed" / "interacoes.csv"
)

ARQUIVO_CLIENTES = (
    RAIZ / "data" / "processed" / "matrizes" / "clientes_ids.npy"
)

ARQUIVO_PRODUTOS = (
    RAIZ / "data" / "processed" / "matrizes" / "produtos_ids.npy"
)

PASTA_VERSOES = (
    RAIZ / "data" / "processed" / "modelos" / "versoes"
)

COLUNA_DATA = "event_timestamp"

DATA_CORTE = pd.Timestamp("2026-07-01")

K = 10

PESOS = {
    "view": 1,
    "click": 2,
    "cart": 3,
    "purchase": 5,
}

NOME_EXPERIMENTO = "RecoCommerce_Brasil"

BANCO_MLFLOW = RAIZ / "mlflow.db"


# ============================================================
# 2. VALIDACAO DOS ARQUIVOS
# ============================================================

def validar_arquivos():

    arquivos = [
        ARQUIVO_INTERACOES,
        ARQUIVO_CLIENTES,
        ARQUIVO_PRODUTOS,
    ]

    ausentes = [
        str(arquivo)
        for arquivo in arquivos
        if not arquivo.exists()
    ]

    if ausentes:
        raise FileNotFoundError(
            "Arquivos necessarios nao encontrados:\n"
            + "\n".join(ausentes)
        )


# ============================================================
# 3. PREPARACAO DOS DADOS
# ============================================================

def preparar_dados():

    print("\nCarregando dados...")

    validar_arquivos()

    interacoes = pd.read_csv(ARQUIVO_INTERACOES)

    colunas_obrigatorias = {
        "customer_id",
        "product_id",
        "event_type",
        COLUNA_DATA,
    }

    colunas_ausentes = (
        colunas_obrigatorias - set(interacoes.columns)
    )

    if colunas_ausentes:
        raise ValueError(
            "Colunas ausentes: "
            + ", ".join(sorted(colunas_ausentes))
            + "\nColunas encontradas: "
            + ", ".join(interacoes.columns.astype(str))
        )

    interacoes[COLUNA_DATA] = pd.to_datetime(
        interacoes[COLUNA_DATA],
        errors="coerce",
    )

    datas_invalidas = interacoes[COLUNA_DATA].isna().sum()

    if datas_invalidas > 0:

        print(
            "Linhas removidas por data invalida:",
            datas_invalidas,
        )

        interacoes = interacoes.dropna(
            subset=[COLUNA_DATA]
        ).copy()

    clientes_ids = np.load(
        ARQUIVO_CLIENTES,
        allow_pickle=False,
    )

    produtos_ids = np.load(
        ARQUIVO_PRODUTOS,
        allow_pickle=False,
    )

    mapa_clientes = {
        cliente_id: indice
        for indice, cliente_id in enumerate(clientes_ids)
    }

    mapa_produtos = {
        produto_id: indice
        for indice, produto_id in enumerate(produtos_ids)
    }

    # Separacao temporal dos dados

    treino = interacoes[
        interacoes[COLUNA_DATA] < DATA_CORTE
    ].copy()

    teste = interacoes[
        interacoes[COLUNA_DATA] >= DATA_CORTE
    ].copy()

    # Aplicacao dos pesos das interacoes

    treino["peso"] = treino["event_type"].map(PESOS)

    treino = treino.dropna(
        subset=["peso"]
    ).copy()

    # Conversao dos IDs para indices numericos

    treino["indice_cliente"] = (
        treino["customer_id"].map(mapa_clientes)
    )

    treino["indice_produto"] = (
        treino["product_id"].map(mapa_produtos)
    )

    treino = treino.dropna(
        subset=[
            "indice_cliente",
            "indice_produto",
        ]
    ).copy()

    treino["indice_cliente"] = (
        treino["indice_cliente"].astype(int)
    )

    treino["indice_produto"] = (
        treino["indice_produto"].astype(int)
    )

    # Construcao da matriz esparsa cliente-produto

    matriz_treino = csr_matrix(
        (
            treino["peso"].astype(float).to_numpy(),
            (
                treino["indice_cliente"].to_numpy(),
                treino["indice_produto"].to_numpy(),
            ),
        ),
        shape=(
            len(clientes_ids),
            len(produtos_ids),
        ),
        dtype=np.float64,
    )

    matriz_treino.sum_duplicates()

    print("\n===== PREPARACAO DOS DADOS =====")

    print("Interacoes validas:", len(interacoes))
    print("Interacoes de treino:", len(treino))
    print("Interacoes de teste:", len(teste))

    print("Clientes no cadastro:", len(clientes_ids))
    print("Produtos no catalogo:", len(produtos_ids))

    print("Formato da matriz:", matriz_treino.shape)
    print("Valores nao nulos:", matriz_treino.nnz)

    return (
        treino,
        teste,
        clientes_ids,
        produtos_ids,
        mapa_clientes,
        matriz_treino,
    )


# ============================================================
# 4. RANKING DE POPULARIDADE PARA FALLBACK
# ============================================================

def criar_ranking_popularidade(treino):

    ranking = (
        treino.groupby("product_id")
        .agg(
            pontuacao=("peso", "sum"),
            interacoes=("peso", "count"),
            clientes_unicos=("customer_id", "nunique"),
            compras=(
                "event_type",
                lambda valores: (valores == "purchase").sum(),
            ),
        )
        .reset_index()
        .sort_values(
            [
                "pontuacao",
                "compras",
                "interacoes",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return ranking


# ============================================================
# 5. TREINAMENTO DA SIMILARIDADE ITEM-ITEM
# ============================================================

def treinar_similaridade(matriz_treino):

    print(
        "\nCalculando similaridade item-item por cosseno..."
    )

    similaridade = cosine_similarity(
        matriz_treino.T,
        dense_output=False,
    )

    print("Matriz de similaridade calculada.")
    print("Formato:", similaridade.shape)
    print("Valores nao nulos:", similaridade.nnz)

    return similaridade


# ============================================================
# 6. GERACAO DE RECOMENDACOES
# ============================================================

def gerar_recomendacoes(
    clientes_ids,
    produtos_ids,
    mapa_clientes,
    matriz_treino,
    similaridade,
    ranking,
    quantidade=K,
):

    recomendacoes = []

    produtos_populares = ranking[
        "product_id"
    ].tolist()

    for cliente_id in clientes_ids:

        indice_cliente = mapa_clientes[cliente_id]

        perfil = matriz_treino[indice_cliente]

        resultados = []

        # Produtos ja conhecidos pelo cliente

        produtos_consumidos = {
            produtos_ids[indice]
            for indice in perfil.indices
        }

        # Recomendacao baseada em similaridade

        if perfil.nnz > 0:

            pontuacoes = perfil @ similaridade

            pontuacoes = np.asarray(
                pontuacoes.toarray()
            ).ravel()

            # Evita recomendar produtos ja consumidos

            pontuacoes[perfil.indices] = -np.inf

            indices_ordenados = np.argsort(
                pontuacoes
            )[::-1]

            for indice_produto in indices_ordenados:

                pontuacao = pontuacoes[indice_produto]

                if (
                    not np.isfinite(pontuacao)
                    or pontuacao <= 0
                ):
                    continue

                produto_id = produtos_ids[indice_produto]

                if produto_id in produtos_consumidos:
                    continue

                resultados.append(
                    (
                        produto_id,
                        float(pontuacao),
                        "similaridade_item_item",
                    )
                )

                if len(resultados) >= quantidade:
                    break

        # Fallback de popularidade

        if len(resultados) < quantidade:

            produtos_usados = (
                produtos_consumidos.union(
                    item[0] for item in resultados
                )
            )

            for produto_id in produtos_populares:

                if produto_id in produtos_usados:
                    continue

                resultados.append(
                    (
                        produto_id,
                        0.0,
                        "fallback_popularidade",
                    )
                )

                produtos_usados.add(produto_id)

                if len(resultados) >= quantidade:
                    break

        # Registro das recomendacoes

        for posicao, (
            produto_id,
            pontuacao,
            estrategia,
        ) in enumerate(resultados, start=1):

            recomendacoes.append(
                {
                    "customer_id": cliente_id,
                    "product_id": produto_id,
                    "posicao": posicao,
                    "pontuacao": pontuacao,
                    "estrategia": estrategia,
                }
            )

    return pd.DataFrame(
        recomendacoes,
        columns=[
            "customer_id",
            "product_id",
            "posicao",
            "pontuacao",
            "estrategia",
        ],
    )


# ============================================================
# 7. AVALIACAO TEMPORAL
# ============================================================

def avaliar_modelo(teste, treino, recomendacoes):

    print("\nIniciando avaliacao temporal...")

    # Seleciona somente compras do periodo de teste

    compras_teste = teste[
        teste["event_type"] == "purchase"
    ].copy()

    total_compras_teste = len(compras_teste)

    # Historico de produtos de cada cliente no treino

    produtos_conhecidos = (
        treino.groupby("customer_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    # Verifica se cada produto comprado no teste
    # ainda era desconhecido para aquele cliente

    compras_teste["produto_novo_cliente"] = [
        produto_id not in produtos_conhecidos.get(
            cliente_id,
            set(),
        )
        for cliente_id, produto_id in zip(
            compras_teste["customer_id"],
            compras_teste["product_id"],
        )
    ]

    # Mantem somente compras de produtos novos
    # para o respectivo cliente

    compras_teste = compras_teste[
        compras_teste["produto_novo_cliente"]
    ].copy()

    compras_relevantes = len(compras_teste)

    # Produtos relevantes por cliente

    relevancia = (
        compras_teste.groupby("customer_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    # Recomendacoes organizadas por cliente

    if recomendacoes.empty:

        recomendados_por_cliente = {}

    else:

        recomendados_por_cliente = (
            recomendacoes.sort_values(
                [
                    "customer_id",
                    "posicao",
                ]
            )
            .groupby("customer_id")["product_id"]
            .apply(list)
            .to_dict()
        )

    precisions = []
    recalls = []
    aps = []

    linhas_avaliacao = []

    total_acertos = 0

    # Calculo das metricas por cliente

    for cliente_id, produtos_relevantes in relevancia.items():

        lista_recomendada = (
            recomendados_por_cliente.get(
                cliente_id,
                [],
            )[:K]
        )

        acertos = 0
        soma_precisao = 0.0

        for posicao, produto_id in enumerate(
            lista_recomendada,
            start=1,
        ):

            if produto_id in produtos_relevantes:

                acertos += 1

                soma_precisao += (
                    acertos / posicao
                )

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

        linhas_avaliacao.append(
            {
                "customer_id": cliente_id,
                "produtos_relevantes": len(
                    produtos_relevantes
                ),
                "acertos": acertos,
                "precision_10": precisao,
                "recall_10": recall,
                "map_10": ap,
            }
        )

    # Consolidacao das metricas

    metricas = {
        "precision_10": (
            float(np.mean(precisions))
            if precisions
            else 0.0
        ),

        "recall_10": (
            float(np.mean(recalls))
            if recalls
            else 0.0
        ),

        "map_10": (
            float(np.mean(aps))
            if aps
            else 0.0
        ),

        "clientes_avaliados": int(
            len(precisions)
        ),

        "total_acertos": int(
            total_acertos
        ),

        "compras_teste": int(
            total_compras_teste
        ),

        "compras_relevantes": int(
            compras_relevantes
        ),
    }

    avaliacao = pd.DataFrame(
        linhas_avaliacao
    )

    print("\n===== DIAGNOSTICO DA AVALIACAO =====")

    print(
        "Compras totais no teste:",
        total_compras_teste,
    )

    print(
        "Compras relevantes para avaliacao:",
        compras_relevantes,
    )

    print(
        "Clientes avaliados:",
        metricas["clientes_avaliados"],
    )

    print(
        "Total de acertos:",
        total_acertos,
    )

    return metricas, avaliacao


# ============================================================
# 8. SALVAMENTO DOS ARTEFATOS E METADADOS
# ============================================================

def salvar_versao(
    pasta_versao,
    versao,
    matriz_treino,
    similaridade,
    ranking,
    recomendacoes,
    avaliacao,
    metricas,
    clientes_ids,
    produtos_ids,
):

    pasta_versao.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Matrizes

    save_npz(
        pasta_versao / "matriz_treino.npz",
        matriz_treino,
    )

    save_npz(
        pasta_versao / "matriz_similaridade.npz",
        similaridade,
    )

    # Identificadores

    np.save(
        pasta_versao / "clientes_ids.npy",
        clientes_ids,
    )

    np.save(
        pasta_versao / "produtos_ids.npy",
        produtos_ids,
    )

    # Arquivos CSV

    ranking.to_csv(
        pasta_versao / "ranking_popularidade_fallback.csv",
        index=False,
    )

    recomendacoes.to_csv(
        pasta_versao / "recomendacoes.csv",
        index=False,
    )

    avaliacao.to_csv(
        pasta_versao / "avaliacao.csv",
        index=False,
    )

    # Densidade da matriz

    total_celulas = (
        matriz_treino.shape[0]
        * matriz_treino.shape[1]
    )

    densidade = (
        matriz_treino.nnz / total_celulas
        if total_celulas
        else 0.0
    )

    # Metadados da versao

    metadados = {
        "versao": versao,
        "data_corte": str(DATA_CORTE.date()),
        "k": K,
        "pesos_interacao": PESOS,
        "metodo": "similaridade_item_item_cosseno",
        "fallback": "popularidade_global",
        "fonte_dados": "dados_sinteticos",
        "formato_matriz_treino": list(
            matriz_treino.shape
        ),
        "valores_nao_nulos_matriz_treino": int(
            matriz_treino.nnz
        ),
        "densidade_matriz_treino": float(
            densidade
        ),
        "metricas": metricas,
    }

    with open(
        pasta_versao / "metadados.json",
        "w",
        encoding="utf-8",
    ) as arquivo:

        json.dump(
            metadados,
            arquivo,
            indent=4,
            ensure_ascii=False,
        )

    return densidade


# ============================================================
# 9. REGISTRO NO MLFLOW LOCAL
# ============================================================

def registrar_mlflow(
    versao,
    pasta_versao,
    metricas,
    densidade,
):

    mlflow.set_tracking_uri(
        f"sqlite:///{BANCO_MLFLOW.as_posix()}"
    )

    mlflow.set_experiment(
        NOME_EXPERIMENTO
    )

    with mlflow.start_run(
        run_name=f"Retreinamento_Similaridade_{versao}"
    ):

        # Parametros

        mlflow.log_params(
            {
                "versao": versao,
                "data_corte": str(DATA_CORTE.date()),
                "k": K,
                "metodo": "cosine_similarity_item_item",
                "fallback": "popularidade_global",
                "peso_view": PESOS["view"],
                "peso_click": PESOS["click"],
                "peso_cart": PESOS["cart"],
                "peso_purchase": PESOS["purchase"],
            }
        )

        # Metricas numericas

        metricas_mlflow = {
            chave: float(valor)
            for chave, valor in metricas.items()
            if isinstance(
                valor,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            )
        }

        metricas_mlflow[
            "densidade_matriz_treino"
        ] = float(densidade)

        mlflow.log_metrics(
            metricas_mlflow
        )

        # Tags

        mlflow.set_tags(
            {
                "projeto": "RecoCommerce Brasil",
                "modelo": "Similaridade Item-Item",
                "etapa": "retreinamento",
                "versao": versao,
                "fonte_dados": "dados_sinteticos",
                "ambiente": "local",
            }
        )

        # Artefatos

        mlflow.log_artifacts(
            str(pasta_versao),
            artifact_path="modelo_similaridade",
        )

    print(
        "\nExperimento registrado no MLflow local."
    )


# ============================================================
# 10. EXECUCAO PRINCIPAL
# ============================================================

def main():

    inicio = datetime.now()

    versao = inicio.strftime(
        "%Y%m%d_%H%M%S"
    )

    pasta_versao = (
        PASTA_VERSOES
        / f"similaridade_{versao}"
    )

    # Preparacao dos dados

    (
        treino,
        teste,
        clientes_ids,
        produtos_ids,
        mapa_clientes,
        matriz_treino,
    ) = preparar_dados()

    # Ranking de popularidade

    ranking = criar_ranking_popularidade(
        treino
    )

    # Treinamento da similaridade

    similaridade = treinar_similaridade(
        matriz_treino
    )

    # Geracao das recomendacoes

    recomendacoes = gerar_recomendacoes(
        clientes_ids=clientes_ids,
        produtos_ids=produtos_ids,
        mapa_clientes=mapa_clientes,
        matriz_treino=matriz_treino,
        similaridade=similaridade,
        ranking=ranking,
        quantidade=K,
    )

    # Avaliacao

    metricas, avaliacao = avaliar_modelo(
        teste=teste,
        treino=treino,
        recomendacoes=recomendacoes,
    )

    # Cobertura de clientes

    clientes_recomendados = (
        recomendacoes["customer_id"].nunique()
        if not recomendacoes.empty
        else 0
    )

    # Cobertura do catalogo

    produtos_recomendados = (
        recomendacoes["product_id"].nunique()
        if not recomendacoes.empty
        else 0
    )

    metricas["cobertura_clientes"] = (
        clientes_recomendados / len(clientes_ids)
        if len(clientes_ids)
        else 0.0
    )

    metricas["cobertura_catalogo"] = (
        produtos_recomendados / len(produtos_ids)
        if len(produtos_ids)
        else 0.0
    )

    metricas["clientes_com_recomendacoes"] = int(
        clientes_recomendados
    )

    metricas["produtos_recomendados_unicos"] = int(
        produtos_recomendados
    )

    metricas["interacoes_treino"] = int(
        len(treino)
    )

    metricas["produtos_treino"] = int(
        treino["product_id"].nunique()
    )

    metricas["nnz_matriz"] = int(
        matriz_treino.nnz
    )

    # Salvamento da versao

    densidade = salvar_versao(
        pasta_versao=pasta_versao,
        versao=versao,
        matriz_treino=matriz_treino,
        similaridade=similaridade,
        ranking=ranking,
        recomendacoes=recomendacoes,
        avaliacao=avaliacao,
        metricas=metricas,
        clientes_ids=clientes_ids,
        produtos_ids=produtos_ids,
    )

    # Registro no MLflow

    registrar_mlflow(
        versao=versao,
        pasta_versao=pasta_versao,
        metricas=metricas,
        densidade=densidade,
    )

    duracao = (
        datetime.now() - inicio
    ).total_seconds()

    # Resultado final

    print("\n===== RETREINAMENTO CONCLUIDO =====")

    print("Versao:", versao)

    print(
        "Duracao em segundos:",
        round(duracao, 2),
    )

    print(
        "Pasta da versao:",
        pasta_versao,
    )

    print("\n===== METRICAS =====")

    for chave, valor in metricas.items():

        print(
            f"{chave}: {valor}"
        )

    print(
        "\nDensidade da matriz:",
        densidade,
    )

    print(
        "Experimento registrado no MLflow local."
    )

    print(
        "Arquivos anteriores foram preservados."
    )


# ============================================================
# 11. INICIALIZACAO
# ============================================================

if __name__ == "__main__":
    main()