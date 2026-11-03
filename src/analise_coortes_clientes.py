
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

RAIZ = Path(__file__).resolve().parents[1]

ARQUIVO_INTERACOES = (
    RAIZ / "data" / "processed" / "interacoes.csv"
)

PASTA_SAIDA = (
    RAIZ / "data" / "processed" / "analise_coortes"
)

PASTA_GRAFICOS = PASTA_SAIDA / "graficos"

DATA_CORTE = pd.Timestamp("2026-07-01")

PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CARREGAMENTO E PREPARAÇÃO
# ============================================================

def carregar_dados():
    print("Carregando interações...")

    dados = pd.read_csv(ARQUIVO_INTERACOES)

    dados["event_timestamp"] = pd.to_datetime(
        dados["event_timestamp"],
        errors="coerce"
    )

    dados = dados.dropna(
        subset=["customer_id", "event_timestamp"]
    ).copy()

    dados["mes"] = (
        dados["event_timestamp"].dt.to_period("M")
    )

    referencia = dados[
        dados["event_timestamp"] < DATA_CORTE
    ].copy()

    atual = dados[
        dados["event_timestamp"] >= DATA_CORTE
    ].copy()

    print(f"Interações de referência: {len(referencia):,}")
    print(f"Interações do período atual: {len(atual):,}")

    return referencia, atual


# ============================================================
# 3. MÉTRICAS POR CLIENTE
# ============================================================

def calcular_metricas_clientes(dados, nome_periodo):

    mensal = (
        dados.groupby(
            ["customer_id", "mes"]
        )
        .size()
        .reset_index(name="interacoes")
    )

    clientes = (
        mensal.groupby("customer_id")
        .agg(
            total_interacoes=("interacoes", "sum"),
            meses_ativos=("mes", "nunique")
        )
        .reset_index()
    )

    clientes["media_interacoes_mes_ativo"] = (
        clientes["total_interacoes"]
        / clientes["meses_ativos"]
    )

    clientes["periodo"] = nome_periodo

    return clientes, mensal


# ============================================================
# 4. IDENTIFICAÇÃO DAS COORTES
# ============================================================

def analisar_coortes(referencia, atual):

    metricas_ref, mensal_ref = calcular_metricas_clientes(
        referencia,
        "referencia"
    )

    metricas_atual, mensal_atual = calcular_metricas_clientes(
        atual,
        "atual"
    )

    ids_ref = set(metricas_ref["customer_id"])
    ids_atual = set(metricas_atual["customer_id"])

    recorrentes = ids_ref.intersection(ids_atual)
    novos = ids_atual - ids_ref

    metricas_atual["tipo_cliente"] = (
        metricas_atual["customer_id"]
        .apply(
            lambda x: (
                "Recorrente" if x in recorrentes
                else "Novo"
            )
        )
    )

    print("\n========== COORTES DE CLIENTES ==========")
    print(f"Clientes na referência: {len(ids_ref):,}")
    print(f"Clientes no período atual: {len(ids_atual):,}")
    print(f"Clientes recorrentes: {len(recorrentes):,}")
    print(f"Clientes novos: {len(novos):,}")

    return (
        metricas_ref,
        metricas_atual,
        mensal_ref,
        mensal_atual,
        recorrentes
    )


# ============================================================
# 5. COMPARAÇÃO DOS CLIENTES RECORRENTES
# ============================================================

def comparar_recorrentes(
    metricas_ref,
    metricas_atual,
    recorrentes
):

    ref = metricas_ref[
        metricas_ref["customer_id"].isin(recorrentes)
    ][
        ["customer_id", "media_interacoes_mes_ativo"]
    ].rename(
        columns={
            "media_interacoes_mes_ativo":
            "media_mensal_referencia"
        }
    )

    atual = metricas_atual[
        metricas_atual["customer_id"].isin(recorrentes)
    ][
        ["customer_id", "media_interacoes_mes_ativo"]
    ].rename(
        columns={
            "media_interacoes_mes_ativo":
            "media_mensal_atual"
        }
    )

    comparacao = ref.merge(
        atual,
        on="customer_id",
        how="inner"
    )

    comparacao["diferenca"] = (
        comparacao["media_mensal_atual"]
        - comparacao["media_mensal_referencia"]
    )

    comparacao["variacao_percentual"] = (
        comparacao["diferenca"]
        / comparacao["media_mensal_referencia"].replace(0, pd.NA)
    ) * 100

    comparacao["comportamento"] = (
        comparacao["diferenca"]
        .apply(
            lambda x: (
                "Aumentou" if x > 0
                else "Diminuiu" if x < 0
                else "Sem alteração"
            )
        )
    )

    return comparacao


# ============================================================
# 6. RESUMO DAS COORTES
# ============================================================

def gerar_resumo_coortes(metricas_atual):

    resumo = (
        metricas_atual.groupby("tipo_cliente")
        .agg(
            quantidade_clientes=("customer_id", "nunique"),
            media_interacoes_mes_ativo=(
                "media_interacoes_mes_ativo", "mean"
            ),
            mediana_interacoes_mes_ativo=(
                "media_interacoes_mes_ativo", "median"
            ),
            total_interacoes=("total_interacoes", "sum")
        )
        .reset_index()
    )

    return resumo


# ============================================================
# 7. EVOLUÇÃO MENSAL
# ============================================================

def gerar_evolucao_mensal(referencia, atual):

    dados = pd.concat(
        [referencia, atual],
        ignore_index=True
    )

    mensal = (
        dados.groupby("mes")
        .agg(
            total_interacoes=("customer_id", "size"),
            clientes_ativos=("customer_id", "nunique")
        )
        .reset_index()
    )

    mensal["media_interacoes_cliente"] = (
        mensal["total_interacoes"]
        / mensal["clientes_ativos"]
    )

    mensal["mes"] = mensal["mes"].astype(str)

    return mensal


# ============================================================
# 8. GRÁFICOS
# ============================================================

def gerar_graficos(comparacao, resumo, evolucao):

    # Gráfico 1: comportamento dos clientes recorrentes

    contagem = (
        comparacao["comportamento"]
        .value_counts()
        .reindex(
            ["Aumentou", "Diminuiu", "Sem alteração"],
            fill_value=0
        )
    )

    plt.figure(figsize=(9, 5))
    contagem.plot(kind="bar")
    plt.title("Mudança no comportamento dos clientes recorrentes")
    plt.xlabel("Comportamento")
    plt.ylabel("Quantidade de clientes")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        PASTA_GRAFICOS / "comportamento_recorrentes.png",
        dpi=150
    )
    plt.close()

    # Gráfico 2: média de interações por mês ativo

    plt.figure(figsize=(9, 5))
    resumo.set_index("tipo_cliente")[
        "media_interacoes_mes_ativo"
    ].plot(kind="bar")

    plt.title("Interações médias por mês ativo")
    plt.xlabel("Tipo de cliente")
    plt.ylabel("Média de interações")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        PASTA_GRAFICOS / "comparacao_coortes.png",
        dpi=150
    )
    plt.close()

    # Gráfico 3: evolução mensal

    plt.figure(figsize=(12, 5))
    plt.plot(
        evolucao["mes"],
        evolucao["media_interacoes_cliente"],
        marker="o"
    )

    plt.title("Média mensal de interações por cliente ativo")
    plt.xlabel("Mês")
    plt.ylabel("Interações por cliente ativo")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        PASTA_GRAFICOS / "evolucao_mensal_clientes.png",
        dpi=150
    )
    plt.close()


# ============================================================
# 9. EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    referencia, atual = carregar_dados()

    (
        metricas_ref,
        metricas_atual,
        mensal_ref,
        mensal_atual,
        recorrentes
    ) = analisar_coortes(referencia, atual)

    comparacao = comparar_recorrentes(
        metricas_ref,
        metricas_atual,
        recorrentes
    )

    resumo = gerar_resumo_coortes(metricas_atual)

    evolucao = gerar_evolucao_mensal(
        referencia,
        atual
    )

    # Salvar resultados

    metricas_ref.to_csv(
        PASTA_SAIDA / "metricas_clientes_referencia.csv",
        index=False
    )

    metricas_atual.to_csv(
        PASTA_SAIDA / "metricas_clientes_atual.csv",
        index=False
    )

    comparacao.to_csv(
        PASTA_SAIDA / "comparacao_clientes_recorrentes.csv",
        index=False
    )

    resumo.to_csv(
        PASTA_SAIDA / "resumo_coortes.csv",
        index=False
    )

    evolucao.to_csv(
        PASTA_SAIDA / "evolucao_mensal.csv",
        index=False
    )

    gerar_graficos(
        comparacao,
        resumo,
        evolucao
    )

    print("\n========== RESUMO DAS COORTES ==========")
    print(resumo.to_string(index=False))

    print("\n========== CLIENTES RECORRENTES ==========")

    print(
        comparacao["comportamento"]
        .value_counts()
        .to_string()
    )

    print("\nMédia de interações por mês ativo:")
    print(
        comparacao[
            [
                "media_mensal_referencia",
                "media_mensal_atual"
            ]
        ].mean().round(4)
    )

    print("\nAnálise concluída.")
    print(f"Arquivos salvos em: {PASTA_SAIDA}")


if __name__ == "__main__":
    main()
