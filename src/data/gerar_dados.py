
"""Gera tabelas relacionais para desenvolvimento do RecoCommerce Brasil."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from faker import Faker


# Caminhos do projeto
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CONFIG = ROOT / "config.yaml"


def main():

    # Carrega as configurações do projeto
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    seed = cfg["project"]["random_seed"]

    rng = np.random.default_rng(seed)

    fake = Faker("pt_BR")
    Faker.seed(seed)

    # Parâmetros dos dados
    n_c = cfg["data"]["n_customers"]
    n_p = cfg["data"]["n_products"]
    n_i = cfg["data"]["n_interactions"]
    n_o = cfg["data"]["n_orders"]

    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"]

    categories = cfg["business"]["categories"]
    states = cfg["business"]["states"]

    RAW.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. GERAÇÃO DOS CLIENTES
    # --------------------------------------------------

    customers = pd.DataFrame({
        "customer_id": np.arange(1, n_c + 1),
        "customer_name": [fake.name() for _ in range(n_c)],
        "age": rng.integers(18, 76, n_c),
        "state": rng.choice(states, n_c),
        "signup_date": pd.to_datetime(
            rng.choice(pd.date_range(start, end), n_c)
        ).strftime("%Y-%m-%d"),
        "device": rng.choice(
            ["mobile", "desktop", "tablet"],
            n_c,
            p=[0.72, 0.24, 0.04]
        ),
        "customer_segment": rng.choice(
            ["novo", "recorrente", "premium"],
            n_c,
            p=[0.35, 0.50, 0.15]
        )
    })

    # --------------------------------------------------
    # 2. GERAÇÃO DOS PRODUTOS
    # --------------------------------------------------

    product_categories = rng.choice(categories, n_p)

    products = pd.DataFrame({
        "product_id": np.arange(1, n_p + 1),
        "product_name": [
            f"{cat} {i:04d}"
            for i, cat in enumerate(product_categories, 1)
        ],
        "category": product_categories,
        "brand": [
            f"Marca_{x:03d}"
            for x in rng.integers(1, 151, n_p)
        ],
        "price": np.round(
            np.exp(rng.normal(4.2, 0.75, n_p)), 2
        ),
        "cost": 0.0,
        "rating": np.round(
            np.clip(rng.normal(4.1, 0.55, n_p), 1, 5), 1
        ),
        "stock": rng.integers(0, 250, n_p)
    })

    products["cost"] = (
        products["price"] * rng.uniform(0.45, 0.82, n_p)
    ).round(2)

    # --------------------------------------------------
    # 3. GERAÇÃO DAS INTERAÇÕES
    # --------------------------------------------------

    # Afinidade dos clientes por categorias de produtos
    customer_pref = rng.integers(0, len(categories), n_c)

    interaction_customers = rng.integers(1, n_c + 1, n_i)

    preferred_category = customer_pref[interaction_customers - 1]

    category_for_event = np.where(
        rng.random(n_i) < 0.68,
        preferred_category,
        rng.integers(0, len(categories), n_i)
    )

    product_ids_by_category = {
        idx: products.loc[
            products["category"] == category,
            "product_id"
        ].to_numpy()
        for idx, category in enumerate(categories)
    }

    interaction_products = np.empty(n_i, dtype=int)

    for category_idx, product_ids in product_ids_by_category.items():

        mask = category_for_event == category_idx

        interaction_products[mask] = rng.choice(
            product_ids,
            mask.sum()
        )

    event_types = rng.choice(
        ["view", "click", "cart", "purchase"],
        n_i,
        p=[0.66, 0.22, 0.09, 0.03]
    )

    interactions = pd.DataFrame({
        "interaction_id": np.arange(1, n_i + 1),
        "customer_id": interaction_customers,
        "product_id": interaction_products,
        "event_type": event_types,
        "event_timestamp": pd.to_datetime(
            rng.choice(
                pd.date_range(start, end, freq="min"),
                n_i
            )
        ),
        "session_id": rng.integers(
            1, max(2, n_i // 3), n_i
        ),
        "device": rng.choice(
            ["mobile", "desktop", "tablet"],
            n_i,
            p=[0.72, 0.24, 0.04]
        ),
        "source": rng.choice(
            ["busca", "organico", "campanha", "social"],
            n_i,
            p=[0.38, 0.30, 0.20, 0.12]
        )
    }).sort_values("event_timestamp")

    # --------------------------------------------------
    # 4. GERAÇÃO DOS PEDIDOS
    # --------------------------------------------------

    order_customers = rng.integers(1, n_c + 1, n_o)

    order_dates = pd.to_datetime(
        rng.choice(
            pd.date_range(start, end, freq="h"),
            n_o
        )
    )

    orders = pd.DataFrame({
        "order_id": np.arange(1, n_o + 1),
        "customer_id": order_customers,
        "order_date": order_dates,
        "payment_method": rng.choice(
            [
                "pix",
                "cartao_credito",
                "boleto",
                "carteira_digital"
            ],
            n_o,
            p=[0.43, 0.42, 0.08, 0.07]
        ),
        "shipping_state": rng.choice(states, n_o),
        "status": rng.choice(
            ["entregue", "cancelado", "devolvido"],
            n_o,
            p=[0.92, 0.055, 0.025]
        )
    })

    # --------------------------------------------------
    # 5. GERAÇÃO DOS ITENS DOS PEDIDOS
    # --------------------------------------------------

    n_items = int(n_o * 1.65)

    item_order = rng.choice(
        orders["order_id"],
        n_items,
        replace=True
    )

    item_product = rng.integers(1, n_p + 1, n_items)

    price_lookup = products.set_index("product_id")["price"]

    qty = rng.integers(1, 4, n_items)

    item_price = price_lookup.loc[item_product].to_numpy()

    discount = rng.choice(
        [0, 0.05, 0.10, 0.15, 0.20],
        n_items,
        p=[0.48, 0.18, 0.18, 0.10, 0.06]
    )

    items = pd.DataFrame({
        "order_item_id": np.arange(1, n_items + 1),
        "order_id": item_order,
        "product_id": item_product,
        "quantity": qty,
        "unit_price": item_price,
        "discount_pct": discount,
        "line_total": (
            qty * item_price * (1 - discount)
        ).round(2)
    })

    # --------------------------------------------------
    # 6. EXPORTAÇÃO DOS ARQUIVOS CSV
    # --------------------------------------------------

    datasets = {
        "clientes": customers,
        "produtos": products,
        "interacoes": interactions,
        "pedidos": orders,
        "itens_pedido": items
    }

    for name, df in datasets.items():

        df.to_csv(
            RAW / f"{name}.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print(f"{name}.csv: {len(df):,} registros")

    print(f"\nArquivos gerados em: {RAW}")


if __name__ == "__main__":
    main()

