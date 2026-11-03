import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# ==================================================
# 1. CONFIGURAÇÕES
# ==================================================

RAIZ_PROJETO = Path(__file__).resolve().parents[2]

CAMINHO_INTERACOES = (
    RAIZ_PROJETO / "data" / "processed" / "interacoes.csv"
)

CAMINHO_RESULTADO = (
    RAIZ_PROJETO / "reports" / "tabelas"
    / "comparacao_modelos_recomendacao.csv"
)

DATA_CORTE = pd.Timestamp("2026-07-01")

K = 10

PESOS_EVENTOS = {
    "view": 1,
    "click": 2,
    "cart": 3,
    "purchase": 5
}

print("Configurações carregadas.")
# ==================================================
# 2. CARREGAMENTO E SEPARAÇÃO DOS DADOS
# ==================================================

dados = pd.read_csv(CAMINHO_INTERACOES)

dados["event_timestamp"] = pd.to_datetime(
    dados["event_timestamp"]
)

dados["peso"] = dados["event_type"].map(PESOS_EVENTOS)

dados = dados.dropna(subset=["peso"]).copy()

treino = dados[
    dados["event_timestamp"] < DATA_CORTE
].copy()

teste = dados[
    dados["event_timestamp"] >= DATA_CORTE
].copy()

print("\n===== SEPARAÇÃO DOS DADOS =====")
print("Total de interações:", len(dados))
print("Interações de treinamento:", len(treino))
print("Interações de teste:", len(teste))
print("Clientes no treinamento:", treino["customer_id"].nunique())
print("Clientes no teste:", teste["customer_id"].nunique())
# ==================================================
# 3. DEFINIÇÃO DOS CLIENTES E PRODUTOS RELEVANTES
# ==================================================

historico_clientes = (
    treino.groupby("customer_id")["product_id"]
    .apply(set)
    .to_dict()
)

compras_teste = teste[
    teste["event_type"] == "purchase"
].drop_duplicates(
    subset=["customer_id", "product_id"]
)

relevantes = {}

for cliente, grupo in compras_teste.groupby("customer_id"):

    produtos_conhecidos = historico_clientes.get(
        cliente, set()
    )

    produtos_novos = (
        set(grupo["product_id"]) - produtos_conhecidos
    )

    if produtos_novos:
        relevantes[cliente] = produtos_novos

clientes_avaliados = list(relevantes.keys())

print("\n===== CLIENTES AVALIADOS =====")
print("Clientes com compras relevantes:", len(clientes_avaliados))
print("Total de produtos relevantes:",
      sum(len(produtos) for produtos in relevantes.values()))
# ==================================================
# 4. MATRIZ DE INTERAÇÃO CLIENTE-PRODUTO
# ==================================================

dados_agrupados = (
    treino.groupby(["customer_id", "product_id"])["peso"]
    .sum()
    .reset_index()
)

clientes_ids = sorted(treino["customer_id"].unique())
produtos_ids = sorted(treino["product_id"].unique())

mapa_clientes = {
    cliente: indice
    for indice, cliente in enumerate(clientes_ids)
}

mapa_produtos = {
    produto: indice
    for indice, produto in enumerate(produtos_ids)
}

linhas = dados_agrupados["customer_id"].map(mapa_clientes)
colunas = dados_agrupados["product_id"].map(mapa_produtos)
valores = dados_agrupados["peso"].values

matriz = csr_matrix(
    (valores, (linhas, colunas)),
    shape=(len(clientes_ids), len(produtos_ids))
)

print("\n===== MATRIZ DE INTERAÇÕES =====")
print("Dimensão:", matriz.shape)
print("Interações agregadas:", matriz.nnz)
print("Clientes:", len(clientes_ids))
print("Produtos:", len(produtos_ids))

# ==================================================
# 5. POPULARIDADE E SIMILARIDADE ENTRE PRODUTOS
# ==================================================

# Ranking global de popularidade
ranking_popularidade = (
    dados_agrupados.groupby("product_id")["peso"]
    .sum()
    .sort_values(ascending=False)
)

produtos_populares = ranking_popularidade.index.tolist()

# Similaridade Item-Item usando cosseno
matriz_similaridade = cosine_similarity(
    matriz.T,
    dense_output=True
)

print("\n===== MODELOS PREPARADOS =====")
print("Produtos no ranking:", len(produtos_populares))
print("Dimensão da matriz de similaridade:",
      matriz_similaridade.shape)

# ==================================================
# 6. FUNÇÕES DE RECOMENDAÇÃO
# ==================================================

def recomendar_popularidade(cliente):
    conhecidos = historico_clientes.get(cliente, set())

    recomendados = [
        produto for produto in produtos_populares
        if produto not in conhecidos
    ]

    return recomendados[:K]


def recomendar_similaridade(cliente):
    indice_cliente = mapa_clientes.get(cliente)

    if indice_cliente is None:
        return []

    perfil = matriz.getrow(indice_cliente)

    if perfil.nnz == 0:
        return []

    pontuacoes = (
        perfil @ matriz_similaridade
    ).ravel()

    produtos_conhecidos = historico_clientes.get(
        cliente, set()
    )

    candidatos = []

    for indice, pontuacao in enumerate(pontuacoes):
        produto = produtos_ids[indice]

        if produto not in produtos_conhecidos:
            candidatos.append((produto, pontuacao))

    candidatos.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return [
        produto
        for produto, pontuacao in candidatos[:K]
        if pontuacao > 0
    ]


def recomendar_hibrido(cliente):
    recomendacoes = recomendar_similaridade(cliente)

    if recomendacoes:
        return recomendacoes

    return recomendar_popularidade(cliente)


print("\n===== FUNÇÕES DE RECOMENDAÇÃO CRIADAS =====")
# ==================================================
# 7. FUNÇÃO DE AVALIAÇÃO
# ==================================================

def avaliar_modelo(funcao_recomendacao, nome_modelo):

    precisions = []
    recalls = []
    average_precisions = []

    clientes_com_recomendacoes = 0
    total_recomendados = 0
    produtos_recomendados = set()

    for cliente, produtos_relevantes in relevantes.items():

        recomendados = funcao_recomendacao(cliente)[:K]

        if recomendados:
            clientes_com_recomendacoes += 1

        total_recomendados += len(recomendados)
        produtos_recomendados.update(recomendados)

        acertos = [
            1 if produto in produtos_relevantes else 0
            for produto in recomendados
        ]

        total_acertos = sum(acertos)

        precision = total_acertos / K
        recall = total_acertos / len(produtos_relevantes)

        precisions.append(precision)
        recalls.append(recall)

        soma_precisao = 0
        acertos_encontrados = 0

        for posicao, acerto in enumerate(acertos, start=1):

            if acerto:
                acertos_encontrados += 1
                soma_precisao += acertos_encontrados / posicao

        denominador = min(len(produtos_relevantes), K)

        average_precision = soma_precisao / denominador
        average_precisions.append(average_precision)

    cobertura_clientes = (
        clientes_com_recomendacoes / len(relevantes)
    )

    cobertura_catalogo = (
        len(produtos_recomendados) / len(produtos_ids)
    )

    return {
        "modelo": nome_modelo,
        "Precision@10": np.mean(precisions),
        "Recall@10": np.mean(recalls),
        "MAP@10": np.mean(average_precisions),
        "Cobertura_clientes": cobertura_clientes,
        "Cobertura_catalogo": cobertura_catalogo,
        "Clientes_avaliados": len(relevantes),
        "Clientes_com_recomendacoes": clientes_com_recomendacoes,
        "Produtos_recomendados": len(produtos_recomendados)
    }


print("\n===== FUNÇÃO DE AVALIAÇÃO CRIADA =====")
# ==================================================
# 8. COMPARAÇÃO DOS TRÊS MODELOS
# ==================================================

resultados = []

resultados.append(
    avaliar_modelo(
        recomendar_popularidade,
        "Popularidade Global"
    )
)

resultados.append(
    avaliar_modelo(
        recomendar_similaridade,
        "Similaridade Item-Item"
    )
)

resultados.append(
    avaliar_modelo(
        recomendar_hibrido,
        "Híbrido"
    )
)

comparacao = pd.DataFrame(resultados)

print("\n===== COMPARAÇÃO DOS MODELOS =====")

print(
    comparacao.to_string(
        index=False,
        float_format=lambda valor: f"{valor:.6f}"
    )
)

# ==================================================
# 9. EXPORTAÇÃO DOS RESULTADOS
# ==================================================

CAMINHO_RESULTADO.parent.mkdir(
    parents=True,
    exist_ok=True
)

comparacao.to_csv(
    CAMINHO_RESULTADO,
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivo salvo em:")
print(CAMINHO_RESULTADO)

# ==================================================
# 10. DIAGNÓSTICO DO CLIENTE COLD-START
# ==================================================

cliente_teste = 2360

historico = historico_clientes.get(cliente_teste, set())

compras_cliente = relevantes.get(cliente_teste, set())

recomendacoes_pop = recomendar_popularidade(cliente_teste)
recomendacoes_sim = recomendar_similaridade(cliente_teste)
recomendacoes_hibridas = recomendar_hibrido(cliente_teste)

print("\n===== DIAGNÓSTICO COLD-START =====")
print("Cliente:", cliente_teste)
print("Produtos no histórico de treinamento:", len(historico))
print("Produtos relevantes no teste:", compras_cliente)

print("\nPopularidade:")
print("Recomendações:", recomendacoes_pop)
print("Acertos:", set(recomendacoes_pop) & compras_cliente)

print("\nSimilaridade:")
print("Recomendações:", recomendacoes_sim)
print("Acertos:", set(recomendacoes_sim) & compras_cliente)

print("\nHíbrido:")
print("Recomendações:", recomendacoes_hibridas)
print("Acertos:", set(recomendacoes_hibridas) & compras_cliente)