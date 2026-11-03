
"""Pré-processamento das bases do RecoCommerce Brasil."""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def main():

    PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1. Leitura das bases
    clientes = pd.read_csv(RAW / "clientes.csv")
    produtos = pd.read_csv(RAW / "produtos.csv")
    interacoes = pd.read_csv(RAW / "interacoes.csv")
    pedidos = pd.read_csv(RAW / "pedidos.csv")
    itens = pd.read_csv(RAW / "itens_pedido.csv")

    # 2. Padronização das categorias
    produtos["category"] = (
        produtos["category"]
        .astype(str)
        .str.strip()
        .replace({
            "Eletr├┤nicos": "Eletrônicos",
            "EletrÃ´nicos": "Eletrônicos"
        })
    )

    produtos["product_name"] = (
        produtos["product_name"]
        .astype(str)
        .str.replace(
            "Eletr├┤nicos",
            "Eletrônicos",
            regex=False
        )
        .str.replace(
            "EletrÃ´nicos",
            "Eletrônicos",
            regex=False
        )
    )

    # 3. Conversão das datas
    clientes["signup_date"] = pd.to_datetime(
        clientes["signup_date"],
        errors="coerce"
    )

    interacoes["event_timestamp"] = pd.to_datetime(
        interacoes["event_timestamp"],
        errors="coerce"
    )

    pedidos["order_date"] = pd.to_datetime(
        pedidos["order_date"],
        errors="coerce"
    )

    # 4. Padronização de textos
    clientes["state"] = clientes["state"].str.strip().str.upper()

    pedidos["shipping_state"] = (
        pedidos["shipping_state"].str.strip().str.upper()
    )

    interacoes["event_type"] = (
        interacoes["event_type"].str.strip().str.lower()
    )

    interacoes["source"] = (
        interacoes["source"].str.strip().str.lower()
    )

    # 5. Exportação das bases processadas
    bases = {
        "clientes": clientes,
        "produtos": produtos,
        "interacoes": interacoes,
        "pedidos": pedidos,
        "itens_pedido": itens
    }

    for nome, df in bases.items():

        df.to_csv(
            PROCESSED / f"{nome}.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print(f"{nome}.csv processado: {len(df):,} registros")

    print("\nPré-processamento inicial concluído!")
    print(f"Arquivos salvos em: {PROCESSED}")


if __name__ == "__main__":
    main()
