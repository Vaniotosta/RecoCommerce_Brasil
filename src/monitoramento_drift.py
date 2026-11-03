
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

RAIZ = Path(__file__).resolve().parents[1]

ARQUIVO_INTERACOES = (
    RAIZ / "data" / "processed" / "interacoes.csv"
)

ARQUIVO_PRODUTOS = (
    RAIZ / "data" / "processed" / "produtos.csv"
)

PASTA_SAIDA = (
    RAIZ / "data" / "processed" / "monitoramento_drift"
)

PASTA_GRAFICOS = PASTA_SAIDA / "graficos"

DATA_CORTE = pd.Timestamp("2026-07-01")

PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FUNÇÕES DE CÁLCULO
# ============================================================

def calcular_psi_numerico(referencia, atual, bins=10):

    referencia = pd.Series(referencia).dropna().astype(float)
    atual = pd.Series(atual).dropna().astype(float)

    if referencia.empty or atual.empty:
        return np.nan

    limites = np.unique(
        np.quantile(
            referencia,
            np.linspace(0, 1, bins + 1)
        )
    )

    if len(limites) < 2:
        return 0.0

    limites[0] = -np.inf
    limites[-1] = np.inf

    cont_ref = np.histogram(referencia, bins=limites)[0]
    cont_atual = np.histogram(atual, bins=limites)[0]

    prop_ref = cont_ref / cont_ref.sum()
    prop_atual = cont_atual / cont_atual.sum()

    epsilon = 1e-6

    prop_ref = np.clip(prop_ref, epsilon, None)
    prop_atual = np.clip(prop_atual, epsilon, None)

    psi = np.sum(
        (prop_atual - prop_ref)
        * np.log(prop_atual / prop_ref)
    )

    return float(psi)


def calcular_psi_categorico(referencia, atual):

    referencia = pd.Series(referencia).dropna().astype(str)
    atual = pd.Series(atual).dropna().astype(str)

    if referencia.empty or atual.empty:
        return np.nan

    categorias = sorted(
        set(referencia) | set(atual)
    )

    prop_ref = (
        referencia.value_counts(normalize=True)
        .reindex(categorias, fill_value=0)
    )

    prop_atual = (
        atual.value_counts(normalize=True)
        .reindex(categorias, fill_value=0)
    )

    epsilon = 1e-6

    prop_ref = np.clip(prop_ref, epsilon, None)
    prop_atual = np.clip(prop_atual, epsilon, None)

    psi = np.sum(
        (prop_atual - prop_ref)
        * np.log(prop_atual / prop_ref)
    )

    return float(psi)


def interpretar_psi(valor):

    if pd.isna(valor):
        return "Não calculado"

    if valor < 0.10:
        return "Baixa alteração"

    elif valor < 0.25:
        return "Alteração moderada"

    return "Alteração elevada"


def calcular_quantidade_meses(dados):

    primeiro_mes = dados["event_timestamp"].min().to_period("M")
    ultimo_mes = dados["event_timestamp"].max().to_period("M")

    return (
        ultimo_mes.year - primeiro_mes.year
    ) * 12 + ultimo_mes.month - primeiro_mes.month + 1


def calcular_atividade_mensal_por_cliente(dados, meses):

    atividade = (
        dados.groupby("customer_id")
        .size()
        .astype(float)
        / meses
    )

    atividade.name = "interacoes_mensais_por_cliente_ativo"

    return atividade


# ============================================================
# 3. CARREGAMENTO E PREPARAÇÃO
# ============================================================

def carregar_dados():

    print("Carregando dados...")

    if not ARQUIVO_INTERACOES.exists():
        raise FileNotFoundError(ARQUIVO_INTERACOES)

    if not ARQUIVO_PRODUTOS.exists():
        raise FileNotFoundError(ARQUIVO_PRODUTOS)

    interacoes = pd.read_csv(ARQUIVO_INTERACOES)

    produtos = pd.read_csv(ARQUIVO_PRODUTOS)

    colunas_interacoes = [
        "customer_id",
        "product_id",
        "event_timestamp",
        "event_type",
        "device",
        "source"
    ]

    colunas_produtos = [
        "product_id",
        "category",
        "price"
    ]

    for coluna in colunas_interacoes:
        if coluna not in interacoes.columns:
            raise ValueError(
                f"Coluna ausente nas interações: {coluna}"
            )

    for coluna in colunas_produtos:
        if coluna not in produtos.columns:
            raise ValueError(
                f"Coluna ausente nos produtos: {coluna}"
            )

    interacoes["event_timestamp"] = pd.to_datetime(
        interacoes["event_timestamp"],
        errors="coerce"
    )

    interacoes = interacoes.dropna(
        subset=[
            "event_timestamp",
            "customer_id",
            "product_id"
        ]
    )

    dados = interacoes.merge(
        produtos[
            ["product_id", "category", "price"]
        ],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    dados["price"] = pd.to_numeric(
        dados["price"],
        errors="coerce"
    )

    referencia = dados[
        dados["event_timestamp"] < DATA_CORTE
    ].copy()

    atual = dados[
        dados["event_timestamp"] >= DATA_CORTE
    ].copy()

    if referencia.empty or atual.empty:
        raise ValueError(
            "Um dos períodos está vazio."
        )

    meses_ref = calcular_quantidade_meses(referencia)
    meses_atual = calcular_quantidade_meses(atual)

    print(
        f"Interações de referência: {len(referencia):,}"
    )

    print(
        f"Interações atuais: {len(atual):,}"
    )

    print(
        f"Meses de referência: {meses_ref}"
    )

    print(
        f"Meses atuais: {meses_atual}"
    )

    return (
        referencia,
        atual,
        meses_ref,
        meses_atual
    )


# ============================================================
# 4. MONITORAMENTO DE DATA DRIFT
# ============================================================

def monitorar_variaveis(
    referencia,
    atual,
    meses_ref,
    meses_atual
):

    resultados = []

    # Variáveis categóricas

    categoricas = [
        "event_type",
        "category",
        "device",
        "source"
    ]

    for coluna in categoricas:

        psi = calcular_psi_categorico(
            referencia[coluna],
            atual[coluna]
        )

        resultados.append({
            "variavel": coluna,
            "tipo": "Categórica",
            "psi": psi,
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
            "interpretacao": interpretar_psi(psi)
        })

    # Variáveis numéricas

    numericas = [
        "price"
    ]

    for coluna in numericas:

        ref = referencia[coluna].dropna()
        cur = atual[coluna].dropna()

        psi = calcular_psi_numerico(ref, cur)

        if len(ref) > 0 and len(cur) > 0:
            ks = ks_2samp(ref, cur)

            ks_statistic = float(ks.statistic)
            ks_pvalue = float(ks.pvalue)
        else:
            ks_statistic = np.nan
            ks_pvalue = np.nan

        resultados.append({
            "variavel": coluna,
            "tipo": "Numérica",
            "psi": psi,
            "ks_statistic": ks_statistic,
            "ks_pvalue": ks_pvalue,
            "interpretacao": interpretar_psi(psi)
        })

    # Atividade mensal por cliente ativo
    # Normalização pela quantidade de meses de cada período

    atividade_ref = calcular_atividade_mensal_por_cliente(
        referencia,
        meses_ref
    )

    atividade_atual = calcular_atividade_mensal_por_cliente(
        atual,
        meses_atual
    )

    psi = calcular_psi_numerico(
        atividade_ref,
        atividade_atual
    )

    ks = ks_2samp(
        atividade_ref,
        atividade_atual
    )

    resultados.append({
        "variavel": "interacoes_mensais_por_cliente_ativo",
        "tipo": "Numérica",
        "psi": psi,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "interpretacao": interpretar_psi(psi)
    })

    return pd.DataFrame(resultados), atividade_ref, atividade_atual


# ============================================================
# 5. GERAÇÃO DE GRÁFICOS
# ============================================================

def gerar_graficos(
    referencia,
    atual,
    atividade_ref,
    atividade_atual
):

    categoricas = [
        "event_type",
        "category",
        "device",
        "source"
    ]

    for coluna in categoricas:

        dist_ref = referencia[
            coluna
        ].value_counts(normalize=True)

        dist_atual = atual[
            coluna
        ].value_counts(normalize=True)

        comparacao = pd.concat(
            [dist_ref, dist_atual],
            axis=1
        ).fillna(0)

        comparacao.columns = [
            "Referência",
            "Atual"
        ]

        ax = comparacao.plot(
            kind="bar",
            figsize=(10, 5)
        )

        ax.set_title(
            f"Distribuição de {coluna}"
        )

        ax.set_ylabel("Proporção")
        ax.set_xlabel(coluna)

        plt.xticks(
            rotation=35,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            PASTA_GRAFICOS / f"drift_{coluna}.png",
            dpi=150
        )

        plt.close()

    # Distribuição de preços do catálogo associado

    plt.figure(figsize=(10, 5))

    plt.hist(
        referencia["price"].dropna(),
        bins=30,
        density=True,
        alpha=0.6,
        label="Referência"
    )

    plt.hist(
        atual["price"].dropna(),
        bins=30,
        density=True,
        alpha=0.6,
        label="Atual"
    )

    plt.title(
        "Distribuição do preço atual do catálogo associado"
    )

    plt.xlabel("Preço")
    plt.ylabel("Densidade")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PASTA_GRAFICOS / "drift_price.png",
        dpi=150
    )

    plt.close()

    # Atividade mensal por cliente ativo

    plt.figure(figsize=(10, 5))

    plt.hist(
        atividade_ref,
        bins=30,
        density=True,
        alpha=0.6,
        label="Referência"
    )

    plt.hist(
        atividade_atual,
        bins=30,
        density=True,
        alpha=0.6,
        label="Atual"
    )

    plt.title(
        "Interações médias mensais por cliente ativo"
    )

    plt.xlabel("Interações por cliente / mês")
    plt.ylabel("Densidade")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PASTA_GRAFICOS / "drift_atividade_mensal.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 6. EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    print("\nINICIANDO MONITORAMENTO DE DATA DRIFT\n")

    (
        referencia,
        atual,
        meses_ref,
        meses_atual
    ) = carregar_dados()

    (
        resultados,
        atividade_ref,
        atividade_atual
    ) = monitorar_variaveis(
        referencia,
        atual,
        meses_ref,
        meses_atual
    )

    gerar_graficos(
        referencia,
        atual,
        atividade_ref,
        atividade_atual
    )

    arquivo = PASTA_SAIDA / "resultados_drift.csv"

    resultados.to_csv(
        arquivo,
        index=False
    )

    print("\nRESULTADOS DO MONITORAMENTO\n")

    print(
        resultados.to_string(index=False)
    )

    print("\nDIAGNÓSTICO DA ATIVIDADE DOS CLIENTES")

    print(
        "Média mensal de interações por cliente ativo - referência:",
        round(atividade_ref.mean(), 4)
    )

    print(
        "Média mensal de interações por cliente ativo - atual:",
        round(atividade_atual.mean(), 4)
    )

    print("\nOBSERVAÇÕES METODOLÓGICAS")

    print(
        "- A atividade foi normalizada pela quantidade de meses."
    )

    print(
        "- A análise considera clientes ativos em cada período."
    )

    print(
        "- O teste KS deve ser interpretado com cautela, "
        "pois as observações por cliente não são necessariamente independentes."
    )

    print(
        "- O preço utilizado é o preço atual do catálogo associado "
        "às interações históricas, não o preço histórico da transação."
    )

    print("\nArquivos salvos em:")
    print(PASTA_SAIDA)

    print("\nMonitoramento concluído.")


if __name__ == "__main__":
    main()
