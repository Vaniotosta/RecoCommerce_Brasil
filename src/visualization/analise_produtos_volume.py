
"""Análise de rentabilidade com volume mínimo de vendas."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports" / "tabelas"

VOLUME_MINIMO = 20


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    produtos = pd.read_csv(RAW / "produtos.csv")
    itens = pd.read_csv(RAW / "itens_pedido.csv")

    dados = itens.merge(
        produtos[["product_id", "product_name", "category", "cost"]],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    dados["receita"] = dados["line_total"]
    dados["custo_total"] = dados["quantity"] * dados["cost"]
    dados["lucro_bruto"] = dados["receita"] - dados["custo_total"]

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

    # Filtrar produtos com volume mínimo
    elegiveis = analise[
        analise["unidades_vendidas"] >= VOLUME_MINIMO
    ].copy()

    print("\n===== RESUMO DA ANÁLISE =====")
    print(f"Produtos com vendas: {len(analise)}")
    print(f"Produtos com pelo menos {VOLUME_MINIMO} unidades: {len(elegiveis)}")

    print("\n===== TOP 10 POR LUCRO BRUTO =====")
    print(
        elegiveis.nlargest(10, "lucro_bruto")[
            ["product_id", "product_name", "category",
             "unidades_vendidas", "receita",
             "lucro_bruto", "margem_bruta_pct"]
        ].round(2).to_string(index=False)
    )

    print("\n===== TOP 10 POR MARGEM BRUTA =====")
    print(
        elegiveis.nlargest(10, "margem_bruta_pct")[
            ["product_id", "product_name", "category",
             "unidades_vendidas", "receita",
             "lucro_bruto", "margem_bruta_pct"]
        ].round(2).to_string(index=False)
    )

    print("\n===== PRODUTOS DE ALTA RECEITA E BAIXA MARGEM =====")
    mediana_margem = elegiveis["margem_bruta_pct"].median()

    selecionados = elegiveis[
        elegiveis["margem_bruta_pct"] < mediana_margem
    ].nlargest(10, "receita")

    print(selecionados[
        ["product_id", "product_name", "category",
         "unidades_vendidas", "receita",
         "lucro_bruto", "margem_bruta_pct"]
    ].round(2).to_string(index=False))

    elegiveis.to_csv(
        REPORTS / "rentabilidade_produtos_volume.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nAnálise concluída!")
    print(f"Resultados salvos em: {REPORTS}")


if __name__ == "__main__":
    main()
