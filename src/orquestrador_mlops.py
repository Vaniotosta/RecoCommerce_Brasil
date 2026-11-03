from pathlib import Path
import subprocess
import sys
from datetime import datetime

import mlflow
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ_PROJETO = Path(__file__).resolve().parents[1]

SCRIPT_DRIFT = RAIZ_PROJETO / "src" / "monitoramento_drift.py"
SCRIPT_RETREINAMENTO = (
    RAIZ_PROJETO / "src" / "models" / "retreinar_similaridade.py"
)

BANCO_MLFLOW = RAIZ_PROJETO / "mlflow.db"
PASTA_RESULTADOS_DRIFT = (
    RAIZ_PROJETO / "data" / "processed" / "monitoramento_drift"
)

NOME_EXPERIMENTO = "RecoCommerce_Brasil_MLOps"


# ============================================================
# FUNÇÕES
# ============================================================

def configurar_mlflow():
    """Configura o experimento local do MLflow."""

    mlflow.set_tracking_uri(
        f"sqlite:///{BANCO_MLFLOW.as_posix()}"
    )

    mlflow.set_experiment(NOME_EXPERIMENTO)


def executar_etapa(nome, script):
    """Executa um script Python e interrompe em caso de erro."""

    print("\n" + "=" * 60)
    print(f"EXECUTANDO: {nome}")
    print("=" * 60)

    if not script.exists():
        raise FileNotFoundError(
            f"Script não encontrado: {script}"
        )

    subprocess.run(
        [sys.executable, str(script)],
        cwd=RAIZ_PROJETO,
        check=True
    )

    print(f"\nEtapa concluída: {nome}")


def registrar_resultados_drift():
    """Registra os resultados do monitoramento no MLflow."""

    arquivo = PASTA_RESULTADOS_DRIFT / "resultados_drift.csv"

    if not arquivo.exists():
        print(
            "\nAVISO: arquivo de resultados do drift não encontrado."
        )
        return

    resultados = pd.read_csv(arquivo)

    with mlflow.start_run(
        run_name="Monitoramento_Data_Drift"
    ):
        mlflow.set_tag("etapa", "monitoramento")
        mlflow.set_tag("projeto", "RecoCommerce_Brasil")

        for _, linha in resultados.iterrows():
            variavel = str(linha["variavel"])
            psi = float(linha["psi"])

            nome_metrica = (
                "psi_" + variavel.lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            mlflow.log_metric(nome_metrica[:250], psi)

        mlflow.log_artifact(str(arquivo))

    print("\nResultados do drift registrados no MLflow.")


# ============================================================
# ORQUESTRADOR PRINCIPAL
# ============================================================

def main():

    inicio = datetime.now()

    print("\nRECOMMERCE BRASIL - PIPELINE MLOPS")
    print(f"Início: {inicio}")

    configurar_mlflow()

    # 1. Monitoramento
    executar_etapa(
        "Monitoramento de Data Drift",
        SCRIPT_DRIFT
    )

    # 2. Registro dos resultados
    registrar_resultados_drift()

    # 3. Retreinamento
    executar_etapa(
        "Retreinamento do modelo Item-Item",
        SCRIPT_RETREINAMENTO
    )

    fim = datetime.now()

    print("\n" + "=" * 60)
    print("PIPELINE MLOPS CONCLUÍDO")
    print("=" * 60)
    print(f"Início: {inicio}")
    print(f"Fim: {fim}")
    print(f"Duração: {fim - inicio}")


if __name__ == "__main__":
    main()