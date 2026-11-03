
import pandas as pd
import mlflow

from pathlib import Path
from datetime import datetime

# =====================================================
# 1. CONFIGURACAO DOS CAMINHOS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

ARQUIVO_INTERACOES = (
    BASE_DIR / "data" / "processed" / "interacoes.csv"
)

PASTA_VERSOES = (
    BASE_DIR / "data" / "processed" / "modelos" / "versoes"
)

PASTA_VERSOES.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. CONFIGURACAO DO MLFLOW
# =====================================================

BANCO_MLFLOW = BASE_DIR / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{BANCO_MLFLOW.as_posix()}"
)

mlflow.set_experiment("RecoCommerce_Brasil")

# =====================================================
# 3. IDENTIFICACAO DA VERSAO
# =====================================================

versao = datetime.now().strftime("%Y%m%d_%H%M%S")

nome_arquivo = f"ranking_popularidade_{versao}.csv"

ARQUIVO_SAIDA = PASTA_VERSOES / nome_arquivo

# =====================================================
# 4. CARREGAMENTO DOS DADOS
# =====================================================

dados = pd.read_csv(ARQUIVO_INTERACOES)

dados["event_timestamp"] = pd.to_datetime(
    dados["event_timestamp"]
)

print("Total de interacoes:", len(dados))

# =====================================================
# 5. SEPARACAO TEMPORAL
# =====================================================

DATA_CORTE = pd.Timestamp("2026-07-01")

treino = dados[
    dados["event_timestamp"] < DATA_CORTE
].copy()

if treino.empty:
    raise ValueError("A base de treinamento esta vazia.")

print("Interacoes utilizadas:", len(treino))

# =====================================================
# 6. PONDERACAO DAS INTERACOES
# =====================================================

PESOS = {
    "view": 1,
    "click": 2,
    "cart": 3,
    "purchase": 5
}

treino["peso"] = treino["event_type"].map(PESOS)

if treino["peso"].isna().any():
    raise ValueError("Existem eventos sem peso definido.")

# =====================================================
# 7. TREINAMENTO DO MODELO
# =====================================================

popularidade = (
    treino.groupby("product_id")
    .agg(
        pontuacao_popularidade=("peso", "sum"),
        total_interacoes=("interaction_id", "count"),
        clientes_unicos=("customer_id", "nunique"),
        total_compras=(
            "event_type",
            lambda x: (x == "purchase").sum()
        )
    )
    .sort_values(
        "pontuacao_popularidade",
        ascending=False
    )
    .reset_index()
)

# =====================================================
# 8. SALVAMENTO DA NOVA VERSAO
# =====================================================

popularidade.to_csv(
    ARQUIVO_SAIDA,
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 9. REGISTRO NO MLFLOW
# =====================================================

metricas = {
    "total_interacoes_treino": len(treino),
    "total_produtos": len(popularidade),
    "pontuacao_total": popularidade[
        "pontuacao_popularidade"
    ].sum(),
    "total_compras": treino[
        "event_type"
    ].eq("purchase").sum()
}

parametros = {
    "versao": versao,
    "data_corte": str(DATA_CORTE.date()),
    "peso_view": 1,
    "peso_click": 2,
    "peso_cart": 3,
    "peso_purchase": 5
}

with mlflow.start_run(
    run_name=f"Popularidade_{versao}"
):

    mlflow.log_params(parametros)
    mlflow.log_metrics(metricas)

    mlflow.set_tags({
        "projeto": "RecoCommerce Brasil",
        "modelo": "Popularidade Global",
        "etapa": "retreinamento",
        "versao": versao,
        "fonte_dados": "dados_sinteticos"
    })

# =====================================================
# 10. RESULTADOS
# =====================================================

print("\n===== RETREINAMENTO CONCLUIDO =====")

print("Versao:", versao)

print("Produtos treinados:", len(popularidade))

print("\nTOP 10 PRODUTOS:")

print(
    popularidade.head(10).to_string(index=False)
)

print("\nArquivo salvo em:")
print(ARQUIVO_SAIDA)

print("\nExperimento registrado no MLflow!")
