
import mlflow
from pathlib import Path
from numbers import Real

# ==================================================
# 1. CONFIGURACAO DO PROJETO
# ==================================================

RAIZ_PROJETO = Path(__file__).resolve().parents[2]

BANCO_MLFLOW = RAIZ_PROJETO / "mlflow.db"

NOME_EXPERIMENTO = "RecoCommerce_Brasil"

mlflow.set_tracking_uri(
    f"sqlite:///{BANCO_MLFLOW.as_posix()}"
)

mlflow.set_experiment(NOME_EXPERIMENTO)


# ==================================================
# 2. REGISTRO DOS EXPERIMENTOS
# ==================================================

def registrar_modelo(nome_modelo, metricas, parametros=None):

    with mlflow.start_run(run_name=nome_modelo):

        # Registrar parametros
        if parametros:

            parametros_validos = {
                str(chave): str(valor)
                for chave, valor in parametros.items()
                if valor is not None
            }

            mlflow.log_params(parametros_validos)

        # Corrigir nomes e validar metricas
        metricas_validas = {}

        for chave, valor in metricas.items():

            if isinstance(valor, Real) and not isinstance(valor, bool):

                nome_metrica = (
                    str(chave)
                    .replace("@", "_")
                    .replace("%", "percent")
                    .replace(" ", "_")
                )

                valor_metrica = float(valor)

                if valor_metrica == float("inf") or valor_metrica == float("-inf"):
                    continue

                metricas_validas[nome_metrica] = valor_metrica

        # Registrar metricas
        if metricas_validas:
            mlflow.log_metrics(metricas_validas)

        # Registrar identificacao do projeto
        mlflow.set_tags({
            "projeto": "RecoCommerce Brasil",
            "tipo_experimento": "sistema_recomendacao",
            "modelo": nome_modelo,
            "fonte_dados": "dados_sinteticos"
        })

        print(f"Experimento registrado: {nome_modelo}")

        print("Metricas:", metricas_validas)


# ==================================================
# 3. TESTE DA CONFIGURACAO
# ==================================================

if __name__ == "__main__":

    experimento = mlflow.get_experiment_by_name(
        NOME_EXPERIMENTO
    )

    print("\n===== CONFIGURACAO MLFLOW =====")
    print("Status: configurado")
    print("Experimento:", experimento.name)
    print("ID:", experimento.experiment_id)
    print("Banco:", BANCO_MLFLOW)
