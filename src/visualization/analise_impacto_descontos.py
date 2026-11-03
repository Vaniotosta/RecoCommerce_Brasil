from pathlib import Path
import pandas as pd

# ============================================================
# 1. CAMINHOS
# ============================================================

RAIZ = Path(__file__).resolve().parents[2]

CAMINHO_ITENS = RAIZ / "data" / "processed" / "itens_pedido.csv"
CAMINHO_PRODUTOS = RAIZ / "data" / "processed" / "produtos.csv"
PASTA_SAIDA = RAIZ / "reports" / "tabelas"

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LEITURA DOS DADOS
# ============================================================

itens = pd.read_csv(CAMINHO_ITENS)
produtos = pd.read_csv(CAMINHO_PRODUTOS)

dados = itens.merge(
    produtos[["product_id", "product_name", "category", "cost"]],
    on="product_id",
    how="left"
)

# ============================================================
# 3. CÁLCULOS FINANCEIROS
# ============================================================

dados["receita_sem_desconto"] = (
    dados["quantity"] * dados["unit_price"]
)

dados["valor_desconto"] = (
    dados["receita_sem_desconto"] - dados["line_total"]
)

dados["custo_total"] = (
    dados["quantity"] * dados["cost"]
)

dados["lucro_bruto"] = (
    dados["line_total"] - dados["custo_total"]
)

# ============================================================
# 4. IMPACTO POR PERCENTUAL DE DESCONTO
# ============================================================

analise = (
    dados.groupby("discount_pct")
    .agg(
        itens=("order_item_id", "count"),
        unidades=("quantity", "sum"),
        receita_original=("receita_sem_desconto", "sum"),
        descontos_concedidos=("valor_desconto", "sum"),
        receita_liquida=("line_total", "sum"),
        custo_total=("custo_total", "sum"),
        lucro_bruto=("lucro_bruto", "sum")
    )
    .reset_index()
)

analise["margem_bruta_pct"] = (
    analise["lucro_bruto"] / analise["receita_liquida"] * 100
)

analise["participacao_descontos_pct"] = (
    analise["descontos_concedidos"]
    / analise["receita_original"] * 100
)

# ============================================================
# 5. IMPACTO POR PRODUTO
# ============================================================

por_produto = (
    dados.groupby(
        ["product_id", "product_name", "category"]
    )
    .agg(
        unidades=("quantity", "sum"),
        receita=("line_total", "sum"),
        desconto_total=("valor_desconto", "sum"),
        custo=("custo_total", "sum"),
        lucro_bruto=("lucro_bruto", "sum"),
        desconto_medio=("discount_pct", "mean")
    )
    .reset_index()
)

por_produto["margem_bruta_pct"] = (
    por_produto["lucro_bruto"] / por_produto["receita"] * 100
)

# ============================================================
# 6. RESULTADOS
# ============================================================

print("\n===== IMPACTO DOS DESCONTOS =====")

print(
    analise.round(2).to_string(index=False)
)

print("\n===== PRODUTOS COM MAIOR VALOR DE DESCONTO =====")

print(
    por_produto.nlargest(10, "desconto_total")[
        [
            "product_id",
            "product_name",
            "category",
            "unidades",
            "desconto_total",
            "lucro_bruto",
            "margem_bruta_pct"
        ]
    ].round(2).to_string(index=False)
)

print("\n===== PRODUTOS COM MENOR MARGEM BRUTA =====")

print(
    por_produto.nsmallest(10, "margem_bruta_pct")[
        [
            "product_id",
            "product_name",
            "category",
            "desconto_medio",
            "receita",
            "lucro_bruto",
            "margem_bruta_pct"
        ]
    ].round(2).to_string(index=False)
)

# ============================================================
# 7. EXPORTAÇÃO
# ============================================================

analise.to_csv(
    PASTA_SAIDA / "impacto_descontos.csv",
    index=False,
    encoding="utf-8-sig"
)

por_produto.to_csv(
    PASTA_SAIDA / "rentabilidade_descontos_produtos.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivos salvos em reports/tabelas.")