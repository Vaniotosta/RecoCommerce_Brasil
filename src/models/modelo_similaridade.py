
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. CONFIGURACOES
# ============================================================

SEMENTE = 42
DATA_CORTE = "2026-07-01"
K = 10

PESOS_INTERACAO = {
    "view": 1,
    "click": 2,
    "cart": 3,
    "purchase": 5
}

PASTA_PROJETO = Path(__file__).resolve().parents[2]

CAMINHO_MATRIZ = (
    PASTA_PROJETO
    / "data"
    / "processed"
    / "matrizes"
    / "matriz_cliente_produto.npz"
)

CAMINHO_CLIENTES = (
    PASTA_PROJETO
    / "data"
    / "processed"
    / "matrizes"
    / "clientes_ids.npy"
)

CAMINHO_PRODUTOS = (
    PASTA_PROJETO
    / "data"
    / "processed"
    / "matrizes"
    / "produtos_ids.npy"
)

CAMINHO_INTERACOES = (
    PASTA_PROJETO
    / "data"
    / "processed"
    / "interacoes.csv"
)

PASTA_RELATORIOS = (
    PASTA_PROJETO
    / "reports"
    / "tabelas"
)

PASTA_MODELOS = (
    PASTA_PROJETO
    / "data"
    / "processed"
    / "modelos"
)


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

print("\n===== CARREGAMENTO DOS DADOS =====")

matriz_original = load_npz(CAMINHO_MATRIZ)

clientes_ids = np.load(CAMINHO_CLIENTES)
produtos_ids = np.load(CAMINHO_PRODUTOS)

interacoes = pd.read_csv(CAMINHO_INTERACOES)

interacoes["event_timestamp"] = pd.to_datetime(
    interacoes["event_timestamp"]
)

print("Matriz original:", matriz_original.shape)
print("Clientes:", len(clientes_ids))
print("Produtos:", len(produtos_ids))
print("Interacoes:", len(interacoes))


# ============================================================
# 3. MAPEAMENTO DOS IDENTIFICADORES
# ============================================================

mapa_clientes = {
    cliente_id: indice
    for indice, cliente_id in enumerate(clientes_ids)
}

mapa_produtos = {
    produto_id: indice
    for indice, produto_id in enumerate(produtos_ids)
}


# ============================================================
# 4. SEPARACAO TEMPORAL TREINO E TESTE
# ============================================================

print("\n===== SEPARACAO TREINO E TESTE =====")

data_corte = pd.Timestamp(DATA_CORTE)

dados_treino = interacoes[
    interacoes["event_timestamp"] < data_corte
].copy()

dados_teste = interacoes[
    interacoes["event_timestamp"] >= data_corte
].copy()

print("Interacoes de treinamento:", len(dados_treino))
print("Interacoes de teste:", len(dados_teste))


# ============================================================
# 5. CONSTRUCAO DA MATRIZ DE TREINAMENTO
# ============================================================

print("\n===== MATRIZ DE TREINAMENTO =====")

dados_treino["peso"] = (
    dados_treino["event_type"]
    .map(PESOS_INTERACAO)
)

dados_treino = dados_treino.dropna(
    subset=["peso", "customer_id", "product_id"]
)

dados_treino["indice_cliente"] = (
    dados_treino["customer_id"]
    .map(mapa_clientes)
)

dados_treino["indice_produto"] = (
    dados_treino["product_id"]
    .map(mapa_produtos)
)

dados_treino = dados_treino.dropna(
    subset=["indice_cliente", "indice_produto"]
)

matriz_treino = csr_matrix(
    (
        dados_treino["peso"].astype(float).to_numpy(),
        (
            dados_treino["indice_cliente"].astype(int).to_numpy(),
            dados_treino["indice_produto"].astype(int).to_numpy()
        )
    ),
    shape=(len(clientes_ids), len(produtos_ids))
)

# Soma interacoes repetidas do mesmo cliente com o mesmo produto.
matriz_treino.sum_duplicates()

print("Dimensoes:", matriz_treino.shape)
print("Interacoes ponderadas:", matriz_treino.nnz)

# ============================================================
# 5.1. RANKING DE POPULARIDADE PARA FALLBACK
# ============================================================

popularidade_produtos = (
    dados_treino
    .groupby("product_id")["peso"]
    .sum()
    .sort_values(ascending=False)
)

ranking_popularidade = [
    (int(produto), float(score))
    for produto, score in popularidade_produtos.items()
]

print(
    "Produtos disponíveis no ranking de popularidade:",
    len(ranking_popularidade)
)


# ============================================================
# 6. CALCULO DA SIMILARIDADE ENTRE PRODUTOS
# ============================================================

print("\nCalculando similaridade entre produtos...")

similaridade = cosine_similarity(
    matriz_treino.T,
    dense_output=False
)

print("Matriz de similaridade:", similaridade.shape)
print("Similaridade calculada!")


# ============================================================
# 7. FUNCAO DE RECOMENDACAO
# ============================================================

def recomendar_produtos(cliente_id, quantidade=10):

    if cliente_id not in mapa_clientes:
        return []

    indice_cliente = mapa_clientes[cliente_id]

    perfil_cliente = matriz_treino.getrow(indice_cliente)

    # FALLBACK: cliente sem historico de interacoes
    if perfil_cliente.nnz == 0:

        recomendados = []

        for produto, score in ranking_popularidade:

            recomendados.append((produto, score))

            if len(recomendados) == quantidade:
                break

        return recomendados

    # RECOMENDACAO POR SIMILARIDADE
    pontuacoes = perfil_cliente @ similaridade

    pontuacoes = np.asarray(
        pontuacoes.toarray()
    ).ravel()

    produtos_conhecidos = perfil_cliente.indices

    pontuacoes[produtos_conhecidos] = -np.inf

    indices_ordenados = np.argsort(
        pontuacoes
    )[::-1]

    recomendados = []

    for indice in indices_ordenados:

        pontuacao = pontuacoes[indice]

        if not np.isfinite(pontuacao):
            continue

        if pontuacao <= 0:
            break

        recomendados.append(
            (
                int(produtos_ids[indice]),
                float(pontuacao)
            )
        )

        if len(recomendados) == quantidade:
            break

    return recomendados

    # Impede recomendar produtos que o cliente ja conhece.
    produtos_conhecidos = perfil_cliente.indices

    pontuacoes[produtos_conhecidos] = -np.inf

    indices_ordenados = np.argsort(
        pontuacoes
    )[::-1]

    recomendados = []

    for indice in indices_ordenados:

        pontuacao = pontuacoes[indice]

        if not np.isfinite(pontuacao):
            continue

        if pontuacao <= 0:
            break

        recomendados.append(
            (
                int(produtos_ids[indice]),
                float(pontuacao)
            )
        )

        if len(recomendados) == quantidade:
            break

    return recomendados

# DIAGNOSTICO DO CLIENTE 2360

cliente_teste = 2360

indice = mapa_clientes[cliente_teste]

historico = matriz_treino.getrow(indice)

pontuacoes = historico @ similaridade

pontuacoes = np.asarray(
    pontuacoes.toarray()
).ravel()

print("\n===== INVESTIGACAO DO CLIENTE 2360 =====")
print("Interacoes no treinamento:", historico.nnz)
print("Produtos no historico:", len(historico.indices))
print("Maior pontuacao de similaridade:", pontuacoes.max())
print("Produtos com pontuacao positiva:", np.sum(pontuacoes > 0))


# ============================================================
# 8. EXEMPLO DE RECOMENDACAO
# ============================================================

print("\n===== EXEMPLO DE RECOMENDACAO =====")

cliente_exemplo = 1

recomendacoes_exemplo = recomendar_produtos(
    cliente_exemplo,
    K
)

print("Cliente:", cliente_exemplo)

print(
    "Produtos:",
    [produto for produto, score in recomendacoes_exemplo]
)

print("Quantidade:", len(recomendacoes_exemplo))


# ============================================================
# 9. PREPARACAO DA RELEVANCIA NO TESTE
# ============================================================

print("\n===== PREPARACAO DA AVALIACAO =====")

compras_teste = dados_teste[
    dados_teste["event_type"] == "purchase"
].copy()

compras_teste = compras_teste[
    compras_teste["customer_id"].isin(mapa_clientes)
    & compras_teste["product_id"].isin(mapa_produtos)
]

# Produtos que cada cliente comprou no periodo de teste.
compras_teste = compras_teste[
    ["customer_id", "product_id"]
].drop_duplicates()

# Produtos conhecidos durante o treinamento.
historico_treino = dados_treino[
    ["customer_id", "product_id"]
].drop_duplicates()

historico_por_cliente = (
    historico_treino
    .groupby("customer_id")["product_id"]
    .apply(set)
    .to_dict()
)

# Produtos relevantes: comprados no teste e ainda nao vistos no treino.
relevancia = {}

for cliente, grupo in compras_teste.groupby("customer_id"):

    produtos_comprados = set(grupo["product_id"])

    produtos_conhecidos = historico_por_cliente.get(
        cliente,
        set()
    )

    produtos_relevantes = (
        produtos_comprados - produtos_conhecidos
    )

    if produtos_relevantes:
        relevancia[cliente] = produtos_relevantes

print("Clientes com relevancia:", len(relevancia))


# ============================================================
# 10. AVALIACAO DAS RECOMENDACOES
# ============================================================

print("\n===== AVALIACAO DO MODELO =====")

precisoes = []
recalls = []
mapas = []

registros_recomendacoes = []

clientes_com_recomendacao_ids = set()
clientes_sem_recomendacao = []

for cliente, produtos_relevantes in relevancia.items():

    recomendados = recomendar_produtos(
        cliente,
        K
    )

    # Contagem independente dos clientes atendidos.
    if recomendados:
        clientes_com_recomendacao_ids.add(cliente)
    else:
        clientes_sem_recomendacao.append(cliente)

    produtos_recomendados = [
        produto
        for produto, score in recomendados
    ]

    acertos = [
        produto
        for produto in produtos_recomendados
        if produto in produtos_relevantes
    ]

    # Precision@K
    precisao = len(acertos) / K

    # Recall@K
    recall = (
        len(acertos) / len(produtos_relevantes)
    )

    # Average Precision@K
    soma_precisoes = 0
    quantidade_acertos = 0

    for posicao, produto in enumerate(
        produtos_recomendados,
        start=1
    ):

        if produto in produtos_relevantes:

            quantidade_acertos += 1

            soma_precisoes += (
                quantidade_acertos / posicao
            )

    denominador_ap = min(
        len(produtos_relevantes),
        K
    )

    ap = soma_precisoes / denominador_ap

    precisoes.append(precisao)
    recalls.append(recall)
    mapas.append(ap)

    # Guarda as recomendacoes para exportacao.
    for posicao, (produto, score) in enumerate(
        recomendados,
        start=1
    ):

        registros_recomendacoes.append({
            "customer_id": cliente,
            "product_id": produto,
            "rank": posicao,
            "score": score
        })
# ============================================================
# DIAGNOSTICO DO FALLBACK - CLIENTE 2360
# ============================================================

cliente_diagnostico = 2360

produtos_relevantes = relevancia.get(
    cliente_diagnostico,
    set()
)

produtos_recomendados = {
    produto
    for produto, score in recomendar_produtos(
        cliente_diagnostico,
        K
    )
}

acertos_fallback = (
    produtos_recomendados & produtos_relevantes
)

print("\n===== AVALIACAO DO FALLBACK =====")
print("Cliente:", cliente_diagnostico)
print("Produtos relevantes no teste:", len(produtos_relevantes))
print("Produtos recomendados:", len(produtos_recomendados))
print("Acertos:", len(acertos_fallback))
print("Produtos acertados:", sorted(acertos_fallback))


# ============================================================
# 11. RESULTADOS E DIAGNOSTICO
# ============================================================

precision_at_10 = (
    float(np.mean(precisoes))
    if precisoes else 0
)

recall_at_10 = (
    float(np.mean(recalls))
    if recalls else 0
)

map_at_10 = (
    float(np.mean(mapas))
    if mapas else 0
)

clientes_avaliados = len(relevancia)

clientes_com_recomendacoes = len(
    clientes_com_recomendacao_ids
)

total_recomendacoes = len(
    registros_recomendacoes
)

print("\n===== RESULTADOS DA AVALIACAO =====")

print("Clientes com relevancia:", clientes_avaliados)
print("Clientes avaliados:", len(precisoes))
print(
    "Clientes com recomendacoes:",
    clientes_com_recomendacoes
)
print("Total de recomendacoes:", total_recomendacoes)

print("\n===== DIAGNOSTICO DE RECOMENDACOES =====")

print(
    "Clientes sem recomendacao:",
    clientes_sem_recomendacao
)

print(
    "Quantidade sem recomendacao:",
    len(clientes_sem_recomendacao)
)

print(
    "Quantidade de clientes com recomendacao:",
    len(clientes_com_recomendacao_ids)
)

# Verificacao de consistencia.
assert (
    clientes_avaliados
    == clientes_com_recomendacoes
    + len(clientes_sem_recomendacao)
), "Erro na contagem dos clientes avaliados."

resultados = pd.DataFrame([{
    "modelo": "Similaridade de Produtos",
    "precision_at_10": precision_at_10,
    "recall_at_10": recall_at_10,
    "map_at_10": map_at_10,
    "clientes_avaliados": clientes_avaliados,
    "clientes_com_recomendacoes": clientes_com_recomendacoes,
    "total_recomendacoes": total_recomendacoes
}])

print("\n", resultados.to_string(index=False))


# ============================================================
# 12. EXPORTACAO DOS RESULTADOS
# ============================================================

PASTA_RELATORIOS.mkdir(
    parents=True,
    exist_ok=True
)

PASTA_MODELOS.mkdir(
    parents=True,
    exist_ok=True
)

caminho_avaliacao = (
    PASTA_RELATORIOS
    / "avaliacao_similaridade.csv"
)

caminho_recomendacoes = (
    PASTA_MODELOS
    / "recomendacoes_similaridade.csv"
)

resultados.to_csv(
    caminho_avaliacao,
    index=False,
    encoding="utf-8-sig"
)

df_recomendacoes = pd.DataFrame(
    registros_recomendacoes,
    columns=[
        "customer_id",
        "product_id",
        "rank",
        "score"
    ]
)

df_recomendacoes.to_csv(
    caminho_recomendacoes,
    index=False,
    encoding="utf-8-sig"
)

print("\n===== EXPORTACAO CONCLUIDA =====")

print("Avaliacao:", caminho_avaliacao)
print("Recomendacoes:", caminho_recomendacoes)
