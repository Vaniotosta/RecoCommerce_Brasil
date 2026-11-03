import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Caminhos do projeto
BASE_DIR = Path(__file__).resolve().parents[2]

ARQUIVO = BASE_DIR / "data" / "processed" / "interacoes_ponderadas.csv"
PASTA_FIGURAS = BASE_DIR / "reports" / "figures"
PASTA_TABELAS = BASE_DIR / "reports" / "tabelas"

PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
PASTA_TABELAS.mkdir(parents=True, exist_ok=True)

# Carregamento dos dados
dados = pd.read_csv(ARQUIVO)

# Distribuição de produtos por cliente
produtos_cliente = (
    dados.groupby("customer_id")["product_id"]
    .nunique()
)

# Distribuição de clientes por produto
clientes_produto = (
    dados.groupby("product_id")["customer_id"]
    .nunique()
)

# Estatísticas descritivas
print("\n===== PRODUTOS DIFERENTES POR CLIENTE =====")
print(produtos_cliente.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95]))

print("\n===== CLIENTES DIFERENTES POR PRODUTO =====")
print(clientes_produto.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95]))

# Exportação das tabelas
produtos_cliente.to_csv(
    PASTA_TABELAS / "produtos_por_cliente.csv",
    header=["quantidade_produtos"]
)

clientes_produto.to_csv(
    PASTA_TABELAS / "clientes_por_produto.csv",
    header=["quantidade_clientes"]
)

# Gráfico 1: produtos diferentes por cliente
plt.figure(figsize=(10, 6))
plt.hist(produtos_cliente, bins=30)
plt.title("Distribuição de Produtos Interagidos por Cliente")
plt.xlabel("Quantidade de produtos diferentes")
plt.ylabel("Quantidade de clientes")
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "distribuicao_produtos_por_cliente.png", dpi=300)
plt.close()

# Gráfico 2: clientes diferentes por produto
plt.figure(figsize=(10, 6))
plt.hist(clientes_produto, bins=30)
plt.title("Distribuição de Clientes por Produto")
plt.xlabel("Quantidade de clientes diferentes")
plt.ylabel("Quantidade de produtos")
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "distribuicao_clientes_por_produto.png", dpi=300)
plt.close()

# Gráfico 3: 20 produtos com maior pontuação acumulada
popularidade = (
    dados.groupby("product_id")["pontuacao"]
    .sum()
    .sort_values(ascending=False)
    .head(20)
)

plt.figure(figsize=(12, 7))
popularidade.sort_values().plot(kind="barh")
plt.title("20 Produtos com Maior Pontuação de Interações")
plt.xlabel("Pontuação acumulada")
plt.ylabel("Produto")
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "top_20_produtos_interacoes.png", dpi=300)
plt.close()

print("\nArquivos gerados:")
print("- reports/figures/distribuicao_produtos_por_cliente.png")
print("- reports/figures/distribuicao_clientes_por_produto.png")
print("- reports/figures/top_20_produtos_interacoes.png")
print("- reports/tabelas/produtos_por_cliente.csv")
print("- reports/tabelas/clientes_por_produto.csv")

print("\nAnálise concluída!")