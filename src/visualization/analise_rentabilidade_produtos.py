
"""Análise de rentabilidade por produto - RecoCommerce Brasil."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports" / "tabelas"


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    produtos = pd.read_csv(RAW / "produtos.csv")
    itens = pd.read_csv(RAW / "itens_pedido.csv")

    # Associar vendas aos dados dos produtos
    dados = itens.merge(
        produtos[
            ["product_id", "product_name", "category", "price", "cost"]
        ],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    # Calcular indicadores financeiros
    dados["receita"] = dados["line_total"]
    dados["custo_total"] = dados["quantity"] * dados["cost"]
    dados["lucro_bruto"] = dados["receita"] - dados["custo_total"]

    # Consolidar por produto
    analise = dados.groupby(
        ["product_id", "product_name", "category"]
    ).agg(
        pedidos=("order_id", "nunique"),
        unidades_vendidas=("quantity", "sum"),
        receita=("receita", "sum"),
        custo_total=("custo_total", "sum"),
        lucro_bruto=("lucro_bruto", "sum")
    ).reset_index()

    analise["margem_bruta_pct"] = (
        analise["lucro_bruto"] / analise["receita"] * 100
    )

    analise["ticket_medio"] = (
        analise["receita"] / analise["pedidos"]
    )

    # Ranking dos produtos
    print("\n===== TOP 10 PRODUTOS POR RECEITA =====")
    print(
        analise.nlargest(10, "receita")[
            ["product_id", "product_name", "category",
             "unidades_vendidas", "receita", "lucro_bruto"]
        ].round(2).to_string(index=False)
    )

    print("\n===== TOP 10 PRODUTOS POR LUCRO BRUTO =====")
    print(
        analise.nlargest(10, "lucro_bruto")[
            ["product_id", "product_name", "category",
             "unidades_vendidas", "receita", "lucro_bruto"]
        ].round(2).to_string(index=False)
    )

    print("\n===== TOP 10 PRODUTOS POR MARGEM =====")
    print(
        analise.nlargest(10, "margem_bruta_pct")[
            ["product_id", "product_name", "category",
             "receita", "lucro_bruto", "margem_bruta_pct"]
        ].round(2).to_string(index=False)
    )

    # Exportar resultado completo
    analise = analise.sort_values(
        "lucro_bruto", ascending=False
    )

    analise.to_csv(
        REPORTS / "rentabilidade_por_produto.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nAnálise por produto concluída!")
    print(f"Total de produtos com vendas: {len(analise)}")
    print(f"Arquivo salvo em: {REPORTS}")


if __name__ == "__main__":
    main()
