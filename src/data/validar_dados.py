
"""Valida a qualidade e a integridade das bases do RecoCommerce Brasil."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def main():

    arquivos = {
        "clientes": "clientes.csv",
        "produtos": "produtos.csv",
        "interacoes": "interacoes.csv",
        "pedidos": "pedidos.csv",
        "itens_pedido": "itens_pedido.csv"
    }

    dados = {
        nome: pd.read_csv(RAW / arquivo)
        for nome, arquivo in arquivos.items()
    }

    print("\n========== VOLUME DAS BASES ==========")

    for nome, df in dados.items():
        print(f"{nome}: {len(df):,} registros")

    print("\n========== VALORES AUSENTES ==========")

    for nome, df in dados.items():
        ausentes = int(df.isna().sum().sum())
        print(f"{nome}: {ausentes} valores ausentes")

    print("\n========== REGISTROS DUPLICADOS ==========")

    chaves = {
        "clientes": "customer_id",
        "produtos": "product_id",
        "interacoes": "interaction_id",
        "pedidos": "order_id",
        "itens_pedido": "order_item_id"
    }

    for nome, chave in chaves.items():
        duplicados = dados[nome][chave].duplicated().sum()
        print(f"{nome}: {duplicados} chaves duplicadas")

    print("\n========== INTEGRIDADE REFERENCIAL ==========")

    clientes = set(dados["clientes"]["customer_id"])
    produtos = set(dados["produtos"]["product_id"])
    pedidos = set(dados["pedidos"]["order_id"])

    testes = {
        "Interações com clientes inexistentes":
            (~dados["interacoes"]["customer_id"].isin(clientes)).sum(),

        "Interações com produtos inexistentes":
            (~dados["interacoes"]["product_id"].isin(produtos)).sum(),

        "Pedidos com clientes inexistentes":
            (~dados["pedidos"]["customer_id"].isin(clientes)).sum(),

        "Itens com pedidos inexistentes":
            (~dados["itens_pedido"]["order_id"].isin(pedidos)).sum(),

        "Itens com produtos inexistentes":
            (~dados["itens_pedido"]["product_id"].isin(produtos)).sum()
    }

    for descricao, quantidade in testes.items():
        print(f"{descricao}: {quantidade}")

    print("\n========== VALIDAÇÃO CONCLUÍDA ==========")


if __name__ == "__main__":
    main()
