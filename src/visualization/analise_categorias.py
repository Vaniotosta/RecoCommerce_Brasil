
"""Análise de desempenho das categorias do RecoCommerce Brasil."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports" / "tabelas"


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    # 1. Carregar as bases
    produtos = pd.read_csv(RAW / "produtos.csv")
    interacoes = pd.read_csv(RAW / "interacoes.csv")
    itens = pd.read_csv(RAW / "itens_pedido.csv")

    # 2. Associar interações às categorias
    dados_interacoes = interacoes.merge(
        produtos[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    # 3. Eventos por categoria
    eventos = pd.crosstab(
        dados_interacoes["category"],
        dados_interacoes["event_type"]
    )

    colunas = ["view", "click", "cart", "purchase"]
    eventos = eventos.reindex(columns=colunas, fill_value=0)
    eventos["total_eventos"] = eventos.sum(axis=1)

    eventos = eventos.sort_values(
        "total_eventos", ascending=False
    )

    print("\n===== INTERAÇÕES POR CATEGORIA =====")
    print(eventos.to_string())

    # 4. Vendas por categoria
    dados_vendas = itens.merge(
        produtos[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    vendas = dados_vendas.groupby("category").agg(
        pedidos=("order_id", "nunique"),
        itens_vendidos=("order_item_id", "count"),
        unidades_vendidas=("quantity", "sum"),
        receita=("line_total", "sum")
    )

    vendas["participacao_receita_pct"] = (
        vendas["receita"] / vendas["receita"].sum() * 100
    )

    vendas = vendas.sort_values("receita", ascending=False)

    print("\n===== VENDAS POR CATEGORIA =====")
    print(vendas.round(2).to_string())

    # 5. Exportar resultados
    eventos.to_csv(
        REPORTS / "interacoes_por_categoria.csv",
        encoding="utf-8-sig"
    )

    vendas.to_csv(
        REPORTS / "vendas_por_categoria.csv",
        encoding="utf-8-sig"
    )

    print("\nAnálises concluídas!")
    print(f"Resultados salvos em: {REPORTS}")


if __name__ == "__main__":
    main()
