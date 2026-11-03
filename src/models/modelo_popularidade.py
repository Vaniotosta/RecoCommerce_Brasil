
import pandas as pd
from pathlib import Path

# =====================================================
# 1. CONFIGURAÇÃO DOS CAMINHOS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

ARQUIVO_INTERACOES = (
    BASE_DIR / "data" / "processed" / "interacoes.csv"
)

PASTA_MODELOS = BASE_DIR / "data" / "processed" / "modelos"
PASTA_MODELOS.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. CARREGAMENTO DOS DADOS
# =====================================================

dados = pd.read_csv(ARQUIVO_INTERACOES)

dados["event_timestamp"] = pd.to_datetime(
    dados["event_timestamp"]
)

print("Total de interações:", len(dados))


# =====================================================
# 3. SEPARAÇÃO TEMPORAL
# =====================================================

data_corte = pd.Timestamp("2026-07-01")

treino = dados[
    dados["event_timestamp"] < data_corte
].copy()

print("Interações utilizadas no treinamento:", len(treino))


# =====================================================
# 4. PONDERAÇÃO DAS INTERAÇÕES
# =====================================================

pesos = {
    "view": 1,
    "click": 2,
    "cart": 3,
    "purchase": 5
}

treino["peso"] = treino["event_type"].map(pesos)

if treino["peso"].isna().any():
    raise ValueError("Existem tipos de interação sem peso definido.")


# =====================================================
# 5. CÁLCULO DA POPULARIDADE
# =====================================================

popularidade = (
    treino.groupby("product_id")
    .agg(
        pontuacao_popularidade=("peso", "sum"),
        total_interacoes=("interaction_id", "count"),
        clientes_unicos=("customer_id", "nunique"),
        total_compras=("event_type", lambda x: (x == "purchase").sum())
    )
    .sort_values(
        "pontuacao_popularidade",
        ascending=False
    )
    .reset_index()
)

# =====================================================
# 6. EXPORTAÇÃO DOS RESULTADOS
# =====================================================

arquivo_saida = PASTA_MODELOS / "ranking_popularidade.csv"

popularidade.to_csv(
    arquivo_saida,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 7. APRESENTAÇÃO DOS RESULTADOS
# =====================================================

print("\n===== TOP 20 PRODUTOS POPULARES =====")

print(
    popularidade.head(20).to_string(index=False)
)

print("\n===== RESUMO DO MODELO =====")

print("Produtos ranqueados:", len(popularidade))
print("Pontuação total:", popularidade["pontuacao_popularidade"].sum())
print("Arquivo gerado:", arquivo_saida)

print("\nModelo de popularidade concluído!")
