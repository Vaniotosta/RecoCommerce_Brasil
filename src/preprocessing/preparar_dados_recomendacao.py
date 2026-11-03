
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURAÇÕES E CAMINHOS
# ============================================================

RAIZ = Path(__file__).resolve().parents[2]

CAMINHO_INTERACOES = (
    RAIZ / "data" / "processed" / "interacoes.csv"
)

PASTA_DADOS = RAIZ / "data" / "processed"
PASTA_GRAFICOS = RAIZ / "reports" / "figures"
PASTA_TABELAS = RAIZ / "reports" / "tabelas"

for pasta in [PASTA_DADOS, PASTA_GRAFICOS, PASTA_TABELAS]:
    pasta.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LEITURA DAS INTERAÇÕES
# ============================================================

interacoes = pd.read_csv(CAMINHO_INTERACOES)

print("Total de interações:", len(interacoes))
print("Clientes:", interacoes["customer_id"].nunique())
print("Produtos:", interacoes["product_id"].nunique())

# ============================================================
# 3. PESOS DAS INTERAÇÕES
# ============================================================

pesos = {
    "view": 1,
    "click": 2,
    "cart": 3,
    "purchase": 5
}

interacoes["peso"] = (
    interacoes["event_type"].map(pesos)
)

if interacoes["peso"].isna().any():
    raise ValueError(
        "Existem tipos de interação sem peso definido."
    )

# ============================================================
# 4. AGREGAÇÃO CLIENTE-PRODUTO
# ============================================================

matriz_dados = (
    interacoes.groupby(
        ["customer_id", "product_id"]
    )
    .agg(
        pontuacao=("peso", "sum"),
        total_interacoes=("interaction_id", "count")
    )
    .reset_index()
)

# ============================================================
# 5. MATRIZ DE INTERAÇÕES
# ============================================================

matriz = matriz_dados.pivot(
    index="customer_id",
    columns="product_id",
    values="pontuacao"
).fillna(0)

print("\nDimensões da matriz:", matriz.shape)
print("Clientes com interações:", matriz.shape[0])
print("Produtos com interações:", matriz.shape[1])

# ============================================================
# 6. DENSIDADE DA MATRIZ
# ============================================================

total_celulas = matriz.shape[0] * matriz.shape[1]

celulas_preenchidas = (
    matriz.to_numpy() > 0
).sum()

densidade = (
    celulas_preenchidas / total_celulas
) * 100

print(
    f"\nDensidade da matriz: {densidade:.4f}%"
)

print(
    f"Esparsidade da matriz: {100 - densidade:.4f}%"
)

# ============================================================
# 7. EXPORTAÇÃO DOS DADOS
# ============================================================

matriz_dados.to_csv(
    PASTA_DADOS / "interacoes_ponderadas.csv",
    index=False,
    encoding="utf-8-sig"
)

matriz.to_csv(
    PASTA_DADOS / "matriz_cliente_produto.csv",
    encoding="utf-8-sig"
)

# ============================================================
# 8. GRÁFICO: DISTRIBUIÇÃO DOS TIPOS DE INTERAÇÃO
# ============================================================

contagem = interacoes["event_type"].value_counts()

plt.figure(figsize=(9, 5))
contagem.plot(kind="bar")

plt.title("Distribuição dos Tipos de Interação")
plt.xlabel("Tipo de interação")
plt.ylabel("Quantidade")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    PASTA_GRAFICOS / "distribuicao_interacoes_recomendacao.png",
    dpi=300
)

plt.close()

# ============================================================
# 9. GRÁFICO: DISTRIBUIÇÃO DAS PONTUAÇÕES
# ============================================================

plt.figure(figsize=(9, 5))

matriz_dados["pontuacao"].hist(bins=30)

plt.title("Distribuição das Pontuações Cliente-Produto")
plt.xlabel("Pontuação acumulada")
plt.ylabel("Quantidade de combinações")
plt.tight_layout()

plt.savefig(
    PASTA_GRAFICOS / "distribuicao_pontuacoes_recomendacao.png",
    dpi=300
)

plt.close()

# ============================================================
# 10. RELATÓRIO FINAL
# ============================================================

print("\n===== PREPARAÇÃO CONCLUÍDA =====")

print("Matriz cliente-produto:", matriz.shape)
print("Combinações cliente-produto:", len(matriz_dados))

print("\nArquivos gerados:")
print("- interacoes_ponderadas.csv")
print("- matriz_cliente_produto.csv")

print("\nGráficos gerados:")
print("- distribuicao_interacoes_recomendacao.png")
print("- distribuicao_pontuacoes_recomendacao.png")
