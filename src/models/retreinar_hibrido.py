
import json
from pathlib import Path
from datetime import datetime

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

BANCO_MLFLOW = RAIZ / "mlflow.db"

NOME_EXPERIMENTO = "RecoCommerce_Brasil"

COLUNA_DATA = "event_timestamp"

DATA_CORTE = pd.Timestamp("2026-07-01")

K = 10

PESO_SIMILARIDADE = 0.95
PESO_POPULARIDADE = 0.05

PESOS_EVENTOS = {
    "view": 1,
    "click": 2,
    "cart": 3,
    "purchase": 5,
}


# ============================================================
# 2. PREPARACAO DOS DADOS
# ============================================================

def preparar_dados():

    print("\nCarregando dados...")

    arquivos = [
        ARQUIVO_INTERACOES,
        ARQUIVO_CLIENTES,
        ARQUIVO_PRODUTOS,
    ]

    for arquivo in arquivos:
        if not arquivo.exists():
            raise FileNotFoundError(
                f"Arquivo nao encontrado: {arquivo}"
            )

    dados = pd.read_csv(ARQUIVO_INTERACOES)

    colunas = {
        "customer_id",
        "product_id",
        "event_type",
        COLUNA_DATA,
    }

    faltantes = colunas - set(dados.columns)

    if faltantes:
        raise ValueError(
            f"Colunas ausentes: {faltantes}"
        )

    dados[COLUNA_DATA] = pd.to_datetime(
        dados[COLUNA_DATA],
        errors="coerce",
    )

    dados = dados.dropna(
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
        valor: indice
        for indice, valor in enumerate(clientes_ids)
    }

    mapa_produtos = {
        valor: indice
        for indice, valor in enumerate(produtos_ids)
    }

    treino = dados[
        dados[COLUNA_DATA] < DATA_CORTE
    ].copy()

    teste = dados[
        dados[COLUNA_DATA] >= DATA_CORTE
    ].copy()

    treino["peso"] = treino["event_type"].map(
        PESOS_EVENTOS
    )

    treino = treino.dropna(
        subset=["peso"]
    ).copy()

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

    matriz = csr_matrix(
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

    matriz.sum_duplicates()

    print("\n===== DADOS PREPARADOS =====")
    print("Interacoes:", len(dados))
    print("Treino:", len(treino))
    print("Teste:", len(teste))
    print("Clientes:", len(clientes_ids))
    print("Produtos:", len(produtos_ids))
    print("Matriz:", matriz.shape)
    print("Valores nao nulos:", matriz.nnz)

    return (
        treino,
        teste,
        clientes_ids,
        produtos_ids,
        mapa_clientes,
        matriz,
    )


# ============================================================
# 3. MODELO DE POPULARIDADE
# ============================================================

def criar_popularidade(treino, produtos_ids):

    ranking = (
        treino.groupby("product_id")
        .agg(
            pontuacao=("peso", "sum"),
            interacoes=("peso", "count"),
            clientes_unicos=("customer_id", "nunique"),
            compras=(
                "event_type",
                lambda x: (x == "purchase").sum(),
            ),
        )
        .reset_index()
    )

    # Inclui produtos sem interacoes no periodo de treino
    ranking = (
        pd.DataFrame({"product_id": produtos_ids})
        .merge(
            ranking,
            on="product_id",
            how="left",
        )
        .fillna(0)
    )

    ranking = ranking.sort_values(
        ["pontuacao", "compras", "interacoes"],
        ascending=False,
    ).reset_index(drop=True)

    # Normalizacao Min-Max
    maior = ranking["pontuacao"].max()
    menor = ranking["pontuacao"].min()

    if maior > menor:
        ranking["popularidade_normalizada"] = (
            (ranking["pontuacao"] - menor)
            / (maior - menor)
        )
    else:
        ranking["popularidade_normalizada"] = 0.0

    return ranking


# ============================================================
# 4. SIMILARIDADE ITEM-ITEM
# ============================================================

def treinar_similaridade(matriz):

    print("\nCalculando similaridade Item-Item...")

    similaridade = cosine_similarity(
        matriz.T,
        dense_output=False,
    )

    print("Similaridade calculada.")
    print("Formato:", similaridade.shape)

    return similaridade


# ============================================================
# 5. GERACAO DO MODELO HIBRIDO
# ============================================================

def gerar_recomendacoes(
    clientes_ids,
    produtos_ids,
    mapa_clientes,
    matriz,
    similaridade,
    ranking,
):

    print("\nGerando recomendacoes hibridas...")

    # Pontuacoes de popularidade por indice de produto
    popularidade = (
        ranking.set_index("product_id")
        ["popularidade_normalizada"]
        .reindex(produtos_ids)
        .fillna(0)
        .to_numpy(dtype=float)
    )

    recomendacoes = []

    for contador, cliente_id in enumerate(clientes_ids, start=1):

        indice_cliente = mapa_clientes[cliente_id]

        perfil = matriz[indice_cliente]

        produtos_consumidos = set(
            perfil.indices.tolist()
        )

        # Pontuacao baseada na similaridade
        if perfil.nnz > 0:

            pontuacao_similaridade = np.asarray(
                (perfil @ similaridade).toarray()
            ).ravel()

            maior = pontuacao_similaridade.max()
            menor = pontuacao_similaridade.min()

            if maior > menor:
                pontuacao_similaridade = (
                    (pontuacao_similaridade - menor)
                    / (maior - menor)
                )
            else:
                pontuacao_similaridade[:] = 0.0

        else:
            pontuacao_similaridade = np.zeros(
                len(produtos_ids),
                dtype=float,
            )

        # Combinação das pontuacoes
        pontuacao_hibrida = (
            PESO_SIMILARIDADE * pontuacao_similaridade
            + PESO_POPULARIDADE * popularidade
        )

        # Exclui produtos conhecidos
        if produtos_consumidos:
            pontuacao_hibrida[
                list(produtos_consumidos)
            ] = -np.inf

        indices = np.argsort(
            pontuacao_hibrida
        )[::-1][:K]

        for posicao, indice_produto in enumerate(
            indices,
            start=1,
        ):

            pontuacao = pontuacao_hibrida[
                indice_produto
            ]

            if not np.isfinite(pontuacao):
                continue

            produto_id = produtos_ids[indice_produto]

            recomendacoes.append(
                {
                    "customer_id": cliente_id,
                    "product_id": produto_id,
                    "posicao": posicao,
                    "pontuacao_hibrida": float(pontuacao),
                    "pontuacao_similaridade": float(
                        pontuacao_similaridade[indice_produto]
                    ),
                    "pontuacao_popularidade": float(
                        popularidade[indice_produto]
                    ),
                    "estrategia": (
                        "hibrido_item_item_popularidade"
                        if perfil.nnz > 0
                        else "fallback_popularidade"
                    ),
                }
            )

        if contador % 5000 == 0:
            print(
                f"Clientes processados: {contador}"
            )

    return pd.DataFrame(recomendacoes)


# ============================================================
# 6. AVALIACAO TEMPORAL
# ============================================================

def avaliar_modelo(teste, treino, recomendacoes):

    print("\nAvaliando modelo...")

    compras_teste = teste[
        teste["event_type"] == "purchase"
    ].copy()

    total_compras = len(compras_teste)

    conhecidos = (
        treino.groupby("customer_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    compras_teste = compras_teste[
        [
            produto not in conhecidos.get(cliente, set())
            for cliente, produto in zip(
                compras_teste["customer_id"],
                compras_teste["product_id"],
            )
        ]
    ].copy()

    compras_relevantes = len(compras_teste)

    relevancia = (
        compras_teste.groupby("customer_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    recomendados = (
        recomendacoes.sort_values(
            ["customer_id", "posicao"]
        )
        .groupby("customer_id")["product_id"]
        .apply(list)
        .to_dict()
    )

    precisions = []
    recalls = []
    aps = []

    linhas = []
    total_acertos = 0

    for cliente, produtos_relevantes in relevancia.items():

        lista = recomendados.get(cliente, [])[:K]

        acertos = 0
        soma_precisao = 0.0

        for posicao, produto in enumerate(lista, start=1):

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

        linhas.append(
            {
                "customer_id": cliente,
                "produtos_relevantes": len(
                    produtos_relevantes
                ),
                "acertos": acertos,
                "precision_10": precisao,
                "recall_10": recall,
                "map_10": ap,
            }
        )

    metricas = {
        "precision_10": float(np.mean(precisions)) if precisions else 0.0,
        "recall_10": float(np.mean(recalls)) if recalls else 0.0,
        "map_10": float(np.mean(aps)) if aps else 0.0,
        "clientes_avaliados": len(precisions),
        "total_acertos": total_acertos,
        "compras_teste": total_compras,
        "compras_relevantes": compras_relevantes,
    }

    avaliacao = pd.DataFrame(linhas)

    print("\n===== METRICAS DE AVALIACAO =====")

    for nome, valor in metricas.items():
        print(f"{nome}: {valor}")

    return metricas, avaliacao


# ============================================================
# 7. SALVAMENTO DA VERSAO
# ============================================================

def salvar_versao(
    pasta,
    versao,
    matriz,
    similaridade,
    ranking,
    recomendacoes,
    avaliacao,
    metricas,
    clientes_ids,
    produtos_ids,
):

    pasta.mkdir(
        parents=True,
        exist_ok=False,
    )

    save_npz(
        pasta / "matriz_treino.npz",
        matriz,
    )

    save_npz(
        pasta / "matriz_similaridade.npz",
        similaridade,
    )

    np.save(
        pasta / "clientes_ids.npy",
        clientes_ids,
    )

    np.save(
        pasta / "produtos_ids.npy",
        produtos_ids,
    )

    ranking.to_csv(
        pasta / "ranking_popularidade.csv",
        index=False,
        encoding="utf-8-sig",
    )

    recomendacoes.to_csv(
        pasta / "recomendacoes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    avaliacao.to_csv(
        pasta / "avaliacao.csv",
        index=False,
        encoding="utf-8-sig",
    )

    densidade = matriz.nnz / (
        matriz.shape[0] * matriz.shape[1]
    )

    metadados = {
        "versao": versao,
        "metodo": "hibrido_item_item_popularidade",
        "data_corte": str(DATA_CORTE.date()),
        "k": K,
        "peso_similaridade": PESO_SIMILARIDADE,
        "peso_popularidade": PESO_POPULARIDADE,
        "pesos_eventos": PESOS_EVENTOS,
        "densidade_matriz": float(densidade),
        "metricas": metricas,
        "fonte_dados": "dados_sinteticos",
    }

    with open(
        pasta / "metadados.json",
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
# 8. REGISTRO NO MLFLOW
# ============================================================

def registrar_mlflow(
    versao,
    pasta,
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
        run_name=f"Hibrido_{versao}"
    ):

        mlflow.log_params(
            {
                "versao": versao,
                "data_corte": str(DATA_CORTE.date()),
                "k": K,
                "peso_similaridade": PESO_SIMILARIDADE,
                "peso_popularidade": PESO_POPULARIDADE,
                "metodo": "hibrido_item_item_popularidade",
            }
        )

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
            "densidade_matriz"
        ] = float(densidade)

        mlflow.log_metrics(
            metricas_mlflow
        )

        mlflow.set_tags(
            {
                "projeto": "RecoCommerce Brasil",
                "modelo": "Hibrido Item-Item + Popularidade",
                "etapa": "retreinamento",
                "versao": versao,
                "fonte_dados": "dados_sinteticos",
                "ambiente": "local",
            }
        )

        mlflow.log_artifacts(
            str(pasta),
            artifact_path="modelo_hibrido",
        )

    print("\nRegistro no MLflow concluido.")


# ============================================================
# 9. EXECUCAO PRINCIPAL
# ============================================================

def main():

    inicio = datetime.now()

    versao = inicio.strftime(
        "%Y%m%d_%H%M%S"
    )

    pasta_versao = (
        PASTA_VERSOES / f"hibrido_{versao}"
    )

    (
        treino,
        teste,
        clientes_ids,
        produtos_ids,
        mapa_clientes,
        matriz,
    ) = preparar_dados()

    ranking = criar_popularidade(
        treino,
        produtos_ids,
    )

    similaridade = treinar_similaridade(
        matriz
    )

    recomendacoes = gerar_recomendacoes(
        clientes_ids,
        produtos_ids,
        mapa_clientes,
        matriz,
        similaridade,
        ranking,
    )

    metricas, avaliacao = avaliar_modelo(
        teste,
        treino,
        recomendacoes,
    )

    clientes_recomendados = (
        recomendacoes["customer_id"].nunique()
        if not recomendacoes.empty
        else 0
    )

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

    metricas["clientes_com_recomendacoes"] = (
        clientes_recomendados
    )

    metricas["produtos_recomendados_unicos"] = (
        produtos_recomendados
    )

    metricas["interacoes_treino"] = len(treino)

    metricas["produtos_treino"] = (
        treino["product_id"].nunique()
    )

    metricas["nnz_matriz"] = matriz.nnz

    densidade = salvar_versao(
        pasta=pasta_versao,
        versao=versao,
        matriz=matriz,
        similaridade=similaridade,
        ranking=ranking,
        recomendacoes=recomendacoes,
        avaliacao=avaliacao,
        metricas=metricas,
        clientes_ids=clientes_ids,
        produtos_ids=produtos_ids,
    )

    registrar_mlflow(
        versao,
        pasta_versao,
        metricas,
        densidade,
    )

    duracao = (
        datetime.now() - inicio
    ).total_seconds()

    print("\n===== RETREINAMENTO HIBRIDO CONCLUIDO =====")

    print("Versao:", versao)
    print("Duracao:", round(duracao, 2), "segundos")
    print("Pasta:", pasta_versao)

    print("\n===== METRICAS FINAIS =====")

    for chave, valor in metricas.items():
        print(f"{chave}: {valor}")

    print("\nDensidade da matriz:", densidade)

    print("\nArquivos anteriores preservados.")


if __name__ == "__main__":
    main()
