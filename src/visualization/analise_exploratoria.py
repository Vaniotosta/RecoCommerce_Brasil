
"""Análise exploratória do RecoCommerce Brasil."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "reports" / "figures"


def main():

    FIGURES.mkdir(parents=True, exist_ok=True)

    # 1. Leitura das bases
    produtos = pd.read_csv(RAW / "produtos.csv")
    interacoes = pd.read_csv(RAW / "interacoes.csv")

    sns.set_theme(style="whitegrid")

    # 2. Distribuição de produtos por categoria
    produtos_categoria = (
        produtos["category"]
        .value_counts()
        .sort_values(ascending=False)
    )

    print("\n===== PRODUTOS POR CATEGORIA =====")
    print(produtos_categoria)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=produtos_categoria.values,
        y=produtos_categoria.index
    )

    plt.title("Quantidade de Produtos por Categoria")
    plt.xlabel("Quantidade de Produtos")
    plt.ylabel("Categoria")
    plt.tight_layout()

    plt.savefig(
        FIGURES / "produtos_por_categoria.png",
        dpi=300
    )

    plt.close()

    # 3. Distribuição dos tipos de interação
    interacoes_tipo = (
        interacoes["event_type"]
        .value_counts()
        .sort_values(ascending=False)
    )

    print("\n===== TIPOS DE INTERAÇÃO =====")
    print(interacoes_tipo)

    plt.figure(figsize=(9, 6))

    sns.barplot(
        x=interacoes_tipo.index,
        y=interacoes_tipo.values
    )

    plt.title("Distribuição dos Tipos de Interação")
    plt.xlabel("Tipo de Interação")
    plt.ylabel("Quantidade")
    plt.tight_layout()

    plt.savefig(
        FIGURES / "tipos_interacao.png",
        dpi=300
    )

    plt.close()

    # 4. Produtos mais comprados nas interações
    compras = interacoes[
        interacoes["event_type"] == "purchase"
    ]

    produtos_comprados = (
        compras["product_id"]
        .value_counts()
        .head(10)
    )

    print("\n===== TOP 10 PRODUTOS COMPRADOS =====")
    print(produtos_comprados)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=produtos_comprados.values,
        y=produtos_comprados.index.astype(str)
    )

    plt.title("Top 10 Produtos por Interações de Compra")
    plt.xlabel("Quantidade de Compras Registradas")
    plt.ylabel("ID do Produto")
    plt.tight_layout()

    plt.savefig(
        FIGURES / "top_10_produtos_comprados.png",
        dpi=300
    )

    plt.close()

    print("\nAnálise exploratória inicial concluída!")
    print(f"Gráficos salvos em: {FIGURES}")


if __name__ == "__main__":
    main()
