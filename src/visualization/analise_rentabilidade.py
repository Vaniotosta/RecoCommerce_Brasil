
"""Análise de rentabilidade por categoria - RecoCommerce Brasil."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports" / "tabelas"


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    # 1. Carregar as bases
    produtos = pd.read_csv(RAW / "produtos.csv")
    itens = pd.read_csv(RAW / "itens_pedido.csv")

    # 2. Associar os itens vendidos aos dados dos produtos
    dados = itens.merge(
        produtos[["product_id", "category", "cost"]],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    # 3. Calcular receita e custo dos itens
    dados["custo_total"] = dados["quantity"] * dados["cost"]

    # line_total já considera o desconto aplicado ao item
    dados["receita"] = dados["line_total"]

    # 4. Consolidar os indicadores por categoria
    rentabilidade = dados.groupby("category").agg(
        pedidos=("order_id", "nunique"),
        unidades_vendidas=("quantity", "sum"),
        receita=("receita", "sum"),
        custo_total=("custo_total", "sum")
    )

    # 5. Calcular lucro e margem bruta
    rentabilidade["lucro_bruto"] = (
        rentabilidade["receita"] - rentabilidade["custo_total"]
    )

    rentabilidade["margem_bruta_pct"] = (
        rentabilidade["lucro_bruto"]
        / rentabilidade["receita"] * 100
    )

    # 6. Ticket médio por pedido dentro de cada categoria
    rentabilidade["ticket_medio"] = (
        rentabilidade["receita"] / rentabilidade["pedidos"]
    )

    # 7. Preço médio por unidade vendida
    rentabilidade["preco_medio_unidade"] = (
        rentabilidade["receita"]
        / rentabilidade["unidades_vendidas"]
    )

    rentabilidade = rentabilidade.sort_values(
        "lucro_bruto", ascending=False
    )

    # 8. Exibir resultados
    colunas_monetarias = [
        "receita",
        "custo_total",
        "lucro_bruto",
        "ticket_medio",
        "preco_medio_unidade"
    ]

    print("\n===== ANÁLISE DE RENTABILIDADE =====")
    print(rentabilidade.round(2).to_string())

    print("\n===== DESTAQUES =====")
    print(
        "Maior lucro bruto:",
        rentabilidade["lucro_bruto"].idxmax()
    )
    print(
        "Maior margem bruta:",
        rentabilidade["margem_bruta_pct"].idxmax()
    )

    # 9. Exportar resultado
    rentabilidade.to_csv(
        REPORTS / "rentabilidade_por_categoria.csv",
        encoding="utf-8-sig"
    )

    print("\nAnálise concluída!")
    print(f"Arquivo salvo em: {REPORTS}")


if __name__ == "__main__":
    main()
