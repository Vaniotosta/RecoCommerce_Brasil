
from pathlib import Path
from datetime import datetime
import json

import mlflow
import numpy as np
import pandas as pd

from scipy.sparse import load_npz


# ============================================================
# 1. CONFIGURACOES
# ============================================================

RAIZ = Path(__file__).resolve().parents[2]

VERSAO_ORIGINAL = (
    "similaridade_20261102_090621"
)

PASTA_ORIGINAL = (
    RAIZ / "data" / "processed" / "modelos" /
    "versoes" / VERSAO_ORIGINAL
)

ARQUIVO_INTERACOES = (
    RAIZ / "data" / "processed" / "interacoes.csv"
)

ARQUIVO_PRODUTOS = (
    RAIZ / "data" / "processed" / "produtos.csv"
)

PASTA_VERSOES = (
    RAIZ / "data" / "processed" / "modelos" / "versoes"
)

BANCO_MLFLOW = RAIZ / "mlflow.db"

DATA_CORTE = pd.Timestamp("2026-07-01")

K = 10
CANDIDATOS = 100

LIMITES_CATEGORIA = [5, 4, 3]

NOME_EXPERIMENTO = "RecoCommerce_Brasil"


# ============================================================
# 2. CARREGAMENTO DOS ARTEFATOS ORIGINAIS
# ============================================================

def carregar_dados():

    print("\nCarregando dados e modelo original...")

    interacoes = pd.read_csv(ARQUIVO_INTERACOES)
    produtos = pd.read_csv(ARQUIVO_PRODUTOS)

    interacoes["event_timestamp"] = pd.to_datetime(
        interacoes["event_timestamp"],
        errors="coerce"
    )

    treino = interacoes[
        interacoes["event_timestamp"] < DATA_CORTE
    ].copy()

    teste = interacoes[
        interacoes["event_timestamp"] >= DATA_CORTE
    ].copy()

    clientes_ids = np.load(
        PASTA_ORIGINAL / "clientes_ids.npy",
        allow_pickle=False
    )

    produtos_ids = np.load(
        PASTA_ORIGINAL / "produtos_ids.npy",
        allow_pickle=False
    )

    matriz_treino = load_npz(
        PASTA_ORIGINAL / "matriz_treino.npz"
    )

    similaridade = load_npz(
        PASTA_ORIGINAL / "matriz_similaridade.npz"
    )

    ranking = pd.read_csv(
        PASTA_ORIGINAL / "ranking_popularidade_fallback.csv"
    )

    recomendacoes_originais = pd.read_csv(
        PASTA_ORIGINAL / "recomendacoes.csv"
    )

    mapa_clientes = {
        cliente: indice
        for indice, cliente in enumerate(clientes_ids)
    }

    # Mapeamento dos produtos para suas categorias

    produtos["product_key"] = (
        produtos["product_id"].astype(str)
    )

    mapa_categorias = dict(
        zip(
            produtos["product_key"],
            produtos["category"]
        )
    )

    print("Clientes:", len(clientes_ids))
    print("Produtos:", len(produtos_ids))
    print("Interacoes de treino:", len(treino))
    print("Interacoes de teste:", len(teste))
    print("Matriz de similaridade:", similaridade.shape)

    return (
        treino,
        teste,
        clientes_ids,
        produtos_ids,
        mapa_clientes,
        matriz_treino,
        similaridade,
        ranking,
        recomendacoes_originais,
        mapa_categorias
    )


# ============================================================
# 3. GERACAO DE CANDIDATOS
# ============================================================

def gerar_candidatos(
    clientes_ids,
    produtos_ids,
    mapa_clientes,
    matriz_treino,
    similaridade,
    ranking,
    mapa_categorias
):

    print("\nGerando candidatos para diversificacao...")

    candidatos_por_cliente = {}

    produtos_populares = ranking[
        "product_id"
    ].tolist()

    for numero, cliente_id in enumerate(
        clientes_ids, start=1
    ):

        indice_cliente = mapa_clientes[cliente_id]

        perfil = matriz_treino[indice_cliente]

        produtos_consumidos = {
            str(produtos_ids[indice])
            for indice in perfil.indices
        }

        resultados = []

        produtos_usados = set(produtos_consumidos)

        # Similaridade Item-Item

        if perfil.nnz > 0:

            pontuacoes = perfil @ similaridade

            pontuacoes = np.asarray(
                pontuacoes.toarray()
            ).ravel()

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

                produto_id = produtos_ids[
                    indice_produto
                ]

                produto_key = str(produto_id)

                if produto_key in produtos_usados:
                    continue

                resultados.append({
                    "product_id": produto_id,
                    "pontuacao": float(pontuacao),
                    "estrategia": "similaridade_item_item",
                    "category": mapa_categorias.get(
                        produto_key
                    )
                })

                produtos_usados.add(produto_key)

                if len(resultados) >= CANDIDATOS:
                    break

        # Fallback de popularidade

        if len(resultados) < CANDIDATOS:

            for produto_id in produtos_populares:

                produto_key = str(produto_id)

                if produto_key in produtos_usados:
                    continue

                resultados.append({
                    "product_id": produto_id,
                    "pontuacao": 0.0,
                    "estrategia": "fallback_popularidade",
                    "category": mapa_categorias.get(
                        produto_key
                    )
                })

                produtos_usados.add(produto_key)

                if len(resultados) >= CANDIDATOS:
                    break

        candidatos_por_cliente[cliente_id] = resultados

        if numero % 5000 == 0:
            print(
                f"Clientes processados: {numero}"
            )

    return candidatos_por_cliente


# ============================================================
# 4. APLICACAO DO LIMITE POR CATEGORIA
# ============================================================

def diversificar_recomendacoes(
    candidatos_por_cliente,
    limite
):

    recomendacoes = []

    for cliente_id, candidatos in (
        candidatos_por_cliente.items()
    ):

        selecionados = []
        contagem_categorias = {}
        produtos_selecionados = set()

        # Primeira passagem:
        # respeita o limite por categoria.

        for candidato in candidatos:

            categoria = candidato["category"]

            produto_id = candidato["product_id"]

            if categoria is None or pd.isna(categoria):
                continue

            if produto_id in produtos_selecionados:
                continue

            quantidade_categoria = (
                contagem_categorias.get(categoria, 0)
            )

            if quantidade_categoria >= limite:
                continue

            selecionados.append(candidato)

            produtos_selecionados.add(produto_id)

            contagem_categorias[categoria] = (
                quantidade_categoria + 1
            )

            if len(selecionados) >= K:
                break

        # Segunda passagem:
        # completa a lista caso faltem produtos.
        # Essa etapa pode ultrapassar o limite
        # se nao houver candidatos suficientes.

        if len(selecionados) < K:

            for candidato in candidatos:

                produto_id = candidato["product_id"]

                if produto_id in produtos_selecionados:
                    continue

                selecionados.append(candidato)

                produtos_selecionados.add(produto_id)

                if len(selecionados) >= K:
                    break

        for posicao, candidato in enumerate(
            selecionados, start=1
        ):

            recomendacoes.append({
                "customer_id": cliente_id,
                "product_id": candidato["product_id"],
                "posicao": posicao,
                "pontuacao": candidato["pontuacao"],
                "estrategia": candidato["estrategia"]
            })

    return pd.DataFrame(recomendacoes)


# ============================================================
# 5. AVALIACAO TEMPORAL
# ============================================================

def avaliar_modelo(teste, treino, recomendacoes):

    compras_teste = teste[
        teste["event_type"] == "purchase"
    ].copy()

    produtos_conhecidos = (
        treino.groupby("customer_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    compras_teste["produto_novo_cliente"] = [
        produto not in produtos_conhecidos.get(
            cliente, set()
        )
        for cliente, produto in zip(
            compras_teste["customer_id"],
            compras_teste["product_id"]
        )
    ]

    compras_teste = compras_teste[
        compras_teste["produto_novo_cliente"]
    ]

    relevancia = (
        compras_teste.groupby("customer_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    recomendados_por_cliente = (
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

    for cliente, relevantes in relevancia.items():

        lista = recomendados_por_cliente.get(
            cliente, []
        )[:K]

        acertos = 0
        soma_precisao = 0.0

        for posicao, produto in enumerate(
            lista, start=1
        ):

            if produto in relevantes:

                acertos += 1

                soma_precisao += (
                    acertos / posicao
                )

        precisao = acertos / K

        recall = acertos / len(relevantes)

        ap = (
            soma_precisao
            / min(len(relevantes), K)
        )

        precisions.append(precisao)
        recalls.append(recall)
        aps.append(ap)

        total_acertos += acertos

        linhas.append({
            "customer_id": cliente,
            "acertos": acertos,
            "precision_10": precisao,
            "recall_10": recall,
            "map_10": ap
        })

    metricas = {
        "precision_10": float(np.mean(precisions)),
        "recall_10": float(np.mean(recalls)),
        "map_10": float(np.mean(aps)),
        "clientes_avaliados": len(precisions),
        "total_acertos": total_acertos,
        "compras_relevantes": len(compras_teste)
    }

    return metricas, pd.DataFrame(linhas)


# ============================================================
# 6. METRICAS DE DIVERSIDADE
# ============================================================

def avaliar_diversidade(
    recomendacoes,
    mapa_categorias
):

    dados = recomendacoes.copy()

    dados["category"] = (
        dados["product_id"]
        .astype(str)
        .map(mapa_categorias)
    )

    dados = dados.dropna(
        subset=["category"]
    )

    diversidade = (
        dados.groupby("customer_id")
        .agg(
            categorias_diferentes=(
                "category", "nunique"
            ),
            participacao_dominante=(
                "category",
                lambda x: (
                    x.value_counts().max()
                    / len(x) * 100
                )
            )
        )
        .reset_index()
    )

    metricas = {
        "media_categorias_recomendadas": float(
            diversidade["categorias_diferentes"].mean()
        ),
        "mediana_categorias_recomendadas": float(
            diversidade["categorias_diferentes"].median()
        ),
        "media_participacao_dominante": float(
            diversidade["participacao_dominante"].mean()
        ),
        "clientes_com_uma_categoria_percentual": float(
            (
                diversidade["categorias_diferentes"] == 1
            ).mean() * 100
        )
    }

    return metricas, diversidade


# ============================================================
# 7. SALVAR RESULTADOS
# ============================================================

def salvar_resultados(
    nome,
    recomendacoes,
    avaliacao,
    metricas,
    diversidade
):

    pasta = PASTA_VERSOES / nome

    pasta.mkdir(
        parents=True,
        exist_ok=False
    )

    recomendacoes.to_csv(
        pasta / "recomendacoes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    avaliacao.to_csv(
        pasta / "avaliacao.csv",
        index=False,
        encoding="utf-8-sig"
    )

    diversidade.to_csv(
        pasta / "diversidade.csv",
        index=False,
        encoding="utf-8-sig"
    )

    with open(
        pasta / "metricas.json",
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            metricas,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    return pasta


# ============================================================
# 8. REGISTRO NO MLFLOW
# ============================================================

def registrar_mlflow(nome, metricas):

    mlflow.set_tracking_uri(
        f"sqlite:///{BANCO_MLFLOW.as_posix()}"
    )

    mlflow.set_experiment(
        NOME_EXPERIMENTO
    )

    with mlflow.start_run(
        run_name=nome
    ):

        mlflow.log_params({
            "modelo_base": "similaridade_item_item",
            "versao_original": VERSAO_ORIGINAL,
            "limite_categoria": int(
                nome.split("_limite_")[-1]
            ),
            "candidatos_por_cliente": CANDIDATOS,
            "k": K
        })

        mlflow.log_metrics({
            chave: float(valor)
            for chave, valor in metricas.items()
        })

        mlflow.set_tags({
            "projeto": "RecoCommerce Brasil",
            "modelo": "Item-Item diversificado",
            "etapa": "experimento de diversificacao",
            "ambiente": "local"
        })


# ============================================================
# 9. EXECUCAO PRINCIPAL
# ============================================================

def main():

    arquivos = [
        ARQUIVO_INTERACOES,
        ARQUIVO_PRODUTOS,
        PASTA_ORIGINAL / "matriz_treino.npz",
        PASTA_ORIGINAL / "matriz_similaridade.npz",
        PASTA_ORIGINAL / "clientes_ids.npy",
        PASTA_ORIGINAL / "produtos_ids.npy",
        PASTA_ORIGINAL / "ranking_popularidade_fallback.csv",
        PASTA_ORIGINAL / "recomendacoes.csv"
    ]

    for arquivo in arquivos:

        if not arquivo.exists():
            raise FileNotFoundError(arquivo)

    (
        treino,
        teste,
        clientes_ids,
        produtos_ids,
        mapa_clientes,
        matriz_treino,
        similaridade,
        ranking,
        recomendacoes_originais,
        mapa_categorias
    ) = carregar_dados()

    print("\nAvaliando modelo original...")

    metricas_originais, _ = avaliar_modelo(
        teste,
        treino,
        recomendacoes_originais
    )

    diversidade_original, _ = avaliar_diversidade(
        recomendacoes_originais,
        mapa_categorias
    )

    # Gera candidatos uma unica vez.
    # As duas configuracoes utilizam a mesma lista.

    candidatos = gerar_candidatos(
        clientes_ids,
        produtos_ids,
        mapa_clientes,
        matriz_treino,
        similaridade,
        ranking,
        mapa_categorias
    )

    resultados_comparacao = []

    resultados_comparacao.append({
        "modelo": "Item-Item original",
        **metricas_originais,
        **diversidade_original
    })

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    for limite in LIMITES_CATEGORIA:

        print(
            f"\nAplicando limite de {limite} "
            "produtos por categoria..."
        )

        recomendacoes = diversificar_recomendacoes(
            candidatos,
            limite
        )

        metricas, avaliacao = avaliar_modelo(
            teste,
            treino,
            recomendacoes
        )

        diversidade, dados_diversidade = (
            avaliar_diversidade(
                recomendacoes,
                mapa_categorias
            )
        )

        metricas_completas = {
            **metricas,
            **diversidade
        }

        nome = (
            f"diversificacao_{timestamp}_limite_{limite}"
        )

        pasta = salvar_resultados(
            nome,
            recomendacoes,
            avaliacao,
            metricas_completas,
            dados_diversidade
        )

        registrar_mlflow(
            nome,
            metricas_completas
        )

        resultados_comparacao.append({
            "modelo": f"Item-Item limite {limite}",
            **metricas_completas
        })

        print("\nMetricas da configuracao:")
        for chave, valor in metricas_completas.items():
            print(f"{chave}: {valor}")

        print("Versao salva em:", pasta)

    comparacao = pd.DataFrame(
        resultados_comparacao
    )

    arquivo_comparacao = (
        PASTA_VERSOES
        / f"comparacao_diversificacao_{timestamp}.csv"
    )

    comparacao.to_csv(
        arquivo_comparacao,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== COMPARACAO FINAL =====")
    print(comparacao.to_string(index=False))

    print("\nComparacao salva em:")
    print(arquivo_comparacao)

    print("\nExperimentos registrados no MLflow local.")
    print("Modelo original preservado.")


if __name__ == "__main__":
    main()
