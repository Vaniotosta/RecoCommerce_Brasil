# Dicionário inicial de dados

## clientes
`customer_id` identificador; `age` idade; `state` UF; `signup_date` cadastro; `device` dispositivo; `customer_segment` segmento de simulação.

## produtos
`product_id` identificador; `product_name` nome; `category` categoria; `brand` marca; `price` preço; `cost` custo; `rating` avaliação; `stock` estoque.

## interacoes
`interaction_id` identificador; `customer_id` cliente; `product_id` produto; `event_type` evento; `event_timestamp` data/hora; `session_id` sessão; `device` dispositivo; `source` origem.

## pedidos
`order_id` pedido; `customer_id` cliente; `order_date` data; `payment_method` pagamento; `shipping_state` UF de entrega; `status` situação.

## itens_pedido
`order_item_id` item; `order_id` pedido; `product_id` produto; `quantity` quantidade; `unit_price` preço unitário; `discount_pct` desconto; `line_total` total da linha.
