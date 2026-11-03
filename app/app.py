from pathlib import Path
import json
import hashlib

import pandas as pd
import plotly.express as px
import streamlit as st
from faker import Faker


# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

MODEL = (
    DATA
    / "modelos"
    / "versoes"
    / "similaridade_20261102_090621"
)

st.set_page_config(
    page_title="RecoCommerce Brasil",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. ESTILO VISUAL
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #f4f7fb;
    color: #172b4d;
}

[data-testid="stSidebar"] {
    background-color: #10233f;
}

[data-testid="stSidebar"] * {
    color: #edf4ff;
}

[data-testid="stSidebar"] hr {
    border-color: #304766;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 7px;
    border-radius: 8px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background-color: #1a3557;
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e5eaf2;
    border-radius: 12px;
    padding: 16px;
}

div[data-testid="stMetricLabel"] {
    color: #64748b;
}

div[data-testid="stMetricValue"] {
    color: #142944;
    font-weight: 800;
}

.stButton button {
    border-radius: 8px;
    font-weight: 600;
}

h1, h2, h3 {
    color: #142944;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. CARREGAMENTO DOS DADOS
# ============================================================

@st.cache_data
def carregar_dados():

    arquivo_produtos = DATA / "produtos.csv"
    arquivo_interacoes = DATA / "interacoes.csv"

    if not arquivo_produtos.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {arquivo_produtos}"
        )

    if not arquivo_interacoes.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {arquivo_interacoes}"
        )

    produtos = pd.read_csv(arquivo_produtos)
    interacoes = pd.read_csv(arquivo_interacoes)

    produtos["product_id"] = (
        produtos["product_id"].astype(str)
    )

    interacoes["product_id"] = (
        interacoes["product_id"].astype(str)
    )

    interacoes["customer_id"] = (
        interacoes["customer_id"].astype(str)
    )

    interacoes["event_timestamp"] = pd.to_datetime(
        interacoes["event_timestamp"],
        errors="coerce"
    )

    for coluna in ["price", "cost", "rating", "stock"]:

        if coluna in produtos.columns:

            produtos[coluna] = pd.to_numeric(
                produtos[coluna],
                errors="coerce"
            )

    return produtos, interacoes


@st.cache_data
def carregar_metadados():

    arquivo = MODEL / "metadados.json"

    if not arquivo.exists():
        return {}

    try:
        with open(arquivo, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return {}


@st.cache_data
def carregar_recomendacoes():

    arquivo = MODEL / "recomendacoes.csv"

    if not arquivo.exists():
        return pd.DataFrame()

    return pd.read_csv(arquivo)


try:

    produtos, interacoes = carregar_dados()

except Exception as erro:

    st.error(f"Erro ao carregar os dados: {erro}")
    st.stop()


metadados = carregar_metadados()
recomendacoes = carregar_recomendacoes()


# ============================================================
# 4. FUNÇÕES AUXILIARES
# ============================================================

def formatar_numero(valor):

    return f"{int(valor):,}".replace(",", ".")


def formatar_moeda(valor):

    try:

        return (
            f"R$ {float(valor):,.2f}"
            .replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )

    except (ValueError, TypeError):

        return "Não informado"


@st.cache_data
def gerar_nome_cliente(customer_id):

    """
    Gera um nome fictício estável para cada ID de cliente.
    O nome não representa a identidade real de uma pessoa.
    """

    chave = hashlib.md5(
        str(customer_id).encode("utf-8")
    ).hexdigest()

    gerador = Faker("pt_BR")
    gerador.seed_instance(int(chave[:8], 16))

    return gerador.name()


def gerar_avatar(customer_id):

    """
    Gera uma URL de avatar ilustrativo.
    O avatar é fictício e não representa uma fotografia real.
    """

    chave = hashlib.md5(
        str(customer_id).encode("utf-8")
    ).hexdigest()

    return (
        "https://api.dicebear.com/9.x/avataaars/png"
        f"?seed={chave}&size=160"
    )


def exibir_cabecalho():

    st.title("🛍️ RecoCommerce Brasil")

    st.subheader(
        "Plataforma de inteligência comercial e recomendações"
    )

    st.write(
        "Análise do comportamento dos clientes, "
        "personalização de produtos e inteligência de modelos."
    )

    st.divider()


def exibir_secao(titulo, descricao):

    st.header(titulo)
    st.caption(descricao)


def exibir_cliente(customer_id):

    nome = gerar_nome_cliente(customer_id)
    avatar = gerar_avatar(customer_id)

    coluna_foto, coluna_nome = st.columns([1, 5])

    with coluna_foto:

        st.image(
            avatar,
            width=100
        )

    with coluna_nome:

        st.subheader(nome)

        st.caption(
            f"Cliente fictício · ID {customer_id}"
        )


def configurar_grafico(fig, altura=330):

    fig.update_layout(
        height=altura,
        margin=dict(
            l=10,
            r=10,
            t=45,
            b=10
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans",
            color="#52647a"
        )
    )

    fig.update_xaxes(
        showgrid=False
    )

    fig.update_yaxes(
        gridcolor="#edf1f6",
        zeroline=False
    )

    return fig


def exibir_grafico(fig, altura=330):

    st.plotly_chart(
        configurar_grafico(fig, altura),
        use_container_width=True
    )


# ============================================================
# 5. CATEGORIAS
# ============================================================

if "category" in produtos.columns:

    categorias = sorted(
        produtos["category"]
        .dropna()
        .astype(str)
        .unique()
    )

else:

    categorias = []


# ============================================================
# 6. MENU LATERAL
# ============================================================

with st.sidebar:

    st.title("RecoCommerce")

    st.caption(
        "Inteligência de recomendações"
    )

    st.divider()

    st.subheader("Navegação")

    pagina = st.radio(
        "Selecione uma página",
        [
            "Visão geral",
            "Comportamento dos clientes",
            "Recomendações",
            "Catálogo",
            "Inteligência do modelo"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("Categorias de produtos")

    if categorias:

        for categoria in categorias:

            st.markdown(f"• {categoria}")

    else:

        st.info("Nenhuma categoria encontrada.")

    st.divider()

    st.caption(
        f"Produtos: {formatar_numero(len(produtos))}"
    )

    st.caption(
        f"Interações: {formatar_numero(len(interacoes))}"
    )


# ============================================================
# 7. CABEÇALHO
# ============================================================

exibir_cabecalho()


# ============================================================
# PÁGINA 1 — VISÃO GERAL
# ============================================================

if pagina == "Visão geral":

    exibir_secao(
        "Visão geral do negócio",
        "Indicadores consolidados do catálogo e das interações."
    )

    total_clientes = (
        interacoes["customer_id"].nunique()
    )

    total_produtos = (
        produtos["product_id"].nunique()
    )

    total_interacoes = len(interacoes)

    total_compras = (
        interacoes["event_type"]
        .astype(str)
        .str.lower()
        .eq("purchase")
        .sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Clientes ativos",
        formatar_numero(total_clientes)
    )

    c2.metric(
        "Produtos",
        formatar_numero(total_produtos)
    )

    c3.metric(
        "Interações",
        formatar_numero(total_interacoes)
    )

    c4.metric(
        "Eventos de compra",
        formatar_numero(total_compras)
    )

    st.divider()

    coluna1, coluna2 = st.columns(2)

    with coluna1:

        dados = (
            produtos["category"]
            .value_counts()
            .rename_axis("Categoria")
            .reset_index(name="Produtos")
        )

        fig = px.bar(
            dados.sort_values("Produtos"),
            x="Produtos",
            y="Categoria",
            orientation="h",
            title="Produtos por categoria",
            color_discrete_sequence=["#2563eb"]
        )

        exibir_grafico(fig)

    with coluna2:

        dados = (
            interacoes["event_type"]
            .value_counts()
            .rename_axis("Evento")
            .reset_index(name="Quantidade")
        )

        fig = px.pie(
            dados,
            names="Evento",
            values="Quantidade",
            hole=0.55,
            title="Distribuição dos eventos"
        )

        exibir_grafico(fig)

    dados_mensais = (
        interacoes
        .dropna(subset=["event_timestamp"])
        .assign(
            Mes=lambda df:
            df["event_timestamp"]
            .dt.to_period("M")
            .astype(str)
        )
        .groupby("Mes")
        .size()
        .reset_index(name="Interações")
    )

    fig = px.line(
        dados_mensais,
        x="Mes",
        y="Interações",
        markers=True,
        title="Evolução mensal das interações",
        color_discrete_sequence=["#2563eb"]
    )

    exibir_grafico(fig)


# ============================================================
# PÁGINA 2 — COMPORTAMENTO DOS CLIENTES
# ============================================================

elif pagina == "Comportamento dos clientes":

    exibir_secao(
        "Comportamento dos clientes",
        "Análise dos eventos, dispositivos e canais de interação."
    )

    coluna1, coluna2 = st.columns(2)

    with coluna1:

        dados = (
            interacoes["event_type"]
            .value_counts()
            .rename_axis("Evento")
            .reset_index(name="Quantidade")
        )

        fig = px.bar(
            dados,
            x="Evento",
            y="Quantidade",
            title="Interações por evento",
            color_discrete_sequence=["#2563eb"]
        )

        exibir_grafico(fig)

    with coluna2:

        if "device" in interacoes.columns:

            dados = (
                interacoes["device"]
                .value_counts()
                .rename_axis("Dispositivo")
                .reset_index(name="Quantidade")
            )

            fig = px.pie(
                dados,
                names="Dispositivo",
                values="Quantidade",
                hole=0.5,
                title="Interações por dispositivo"
            )

            exibir_grafico(fig)

    if "source" in interacoes.columns:

        dados = (
            interacoes["source"]
            .value_counts()
            .rename_axis("Origem")
            .reset_index(name="Quantidade")
        )

        fig = px.bar(
            dados,
            x="Origem",
            y="Quantidade",
            title="Origem das interações",
            color_discrete_sequence=["#3b82f6"]
        )

        exibir_grafico(fig)

    st.divider()

    st.subheader("Clientes com maior atividade")

    atividade = (
        interacoes
        .groupby("customer_id")
        .size()
        .sort_values(ascending=False)
        .head(15)
        .rename("Interações")
        .reset_index()
    )

    atividade["Nome do cliente"] = (
        atividade["customer_id"]
        .apply(gerar_nome_cliente)
    )

    st.dataframe(
        atividade[
            [
                "Nome do cliente",
                "customer_id",
                "Interações"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PÁGINA 3 — RECOMENDAÇÕES
# ============================================================

elif pagina == "Recomendações":

    exibir_secao(
        "Recomendações personalizadas",
        "Selecione um cliente para consultar os produtos recomendados."
    )

    if recomendacoes.empty:

        st.warning(
            "O arquivo recomendacoes.csv não foi encontrado "
            "ou está vazio."
        )

    else:

        coluna_cliente = next(
            (
                coluna
                for coluna in [
                    "customer_id",
                    "cliente_id",
                    "user_id"
                ]
                if coluna in recomendacoes.columns
            ),
            None
        )

        coluna_produto = next(
            (
                coluna
                for coluna in [
                    "product_id",
                    "produto_id",
                    "recommended_product_id"
                ]
                if coluna in recomendacoes.columns
            ),
            None
        )

        coluna_rank = next(
            (
                coluna
                for coluna in [
                    "rank",
                    "position",
                    "posicao"
                ]
                if coluna in recomendacoes.columns
            ),
            None
        )

        if coluna_cliente is None or coluna_produto is None:

            st.error(
                "As colunas de cliente e produto não foram identificadas."
            )

        else:

            recomendacoes[coluna_cliente] = (
                recomendacoes[coluna_cliente].astype(str)
            )

            clientes = sorted(
                recomendacoes[coluna_cliente]
                .dropna()
                .unique()
            )

            cliente_selecionado = st.selectbox(
                "Escolha o cliente",
                clientes,
                format_func=lambda x:
                gerar_nome_cliente(x)
            )

            st.divider()

            exibir_cliente(cliente_selecionado)

            resultado = recomendacoes[
                recomendacoes[coluna_cliente]
                == cliente_selecionado
            ].copy()

            if coluna_rank:

                resultado = resultado.sort_values(
                    coluna_rank
                )

            resultado[coluna_produto] = (
                resultado[coluna_produto].astype(str)
            )

            resultado = resultado.merge(
                produtos,
                left_on=coluna_produto,
                right_on="product_id",
                how="left"
            )

            st.subheader(
                "Produtos recomendados"
            )

            if resultado.empty:

                st.info(
                    "Não existem recomendações para este cliente."
                )

            else:

                for inicio in range(
                    0,
                    min(len(resultado), 10),
                    2
                ):

                    colunas = st.columns(2)

                    for coluna, (_, produto) in zip(
                        colunas,
                        resultado.iloc[inicio:inicio + 2].iterrows()
                    ):

                        with coluna:

                            nome = produto.get(
                                "product_name",
                                "Produto"
                            )

                            categoria = produto.get(
                                "category",
                                "Categoria não informada"
                            )

                            preco = formatar_moeda(
                                produto.get("price")
                            )

                            st.container(border=True)

                            st.markdown(
                                f"**{nome}**"
                            )

                            st.caption(
                                categoria
                            )

                            st.markdown(
                                f"### {preco}"
                            )

                st.subheader(
                    "Detalhamento das recomendações"
                )

                st.dataframe(
                    resultado,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# PÁGINA 4 — CATÁLOGO
# ============================================================

elif pagina == "Catálogo":

    exibir_secao(
        "Catálogo de produtos",
        "Pesquise produtos e filtre por categoria."
    )

    coluna1, coluna2 = st.columns([1, 2])

    with coluna1:

        categoria_selecionada = st.selectbox(
            "Categoria do produto",
            ["Todas"] + categorias
        )

    with coluna2:

        busca = st.text_input(
            "Pesquisar produto",
            placeholder="Digite nome, marca ou código"
        )

    catalogo = produtos.copy()

    if categoria_selecionada != "Todas":

        catalogo = catalogo[
            catalogo["category"].astype(str)
            == categoria_selecionada
        ]

    if busca.strip():

        mascara = pd.Series(
            False,
            index=catalogo.index
        )

        for campo in [
            "product_name",
            "brand",
            "product_id"
        ]:

            if campo in catalogo.columns:

                mascara |= (
                    catalogo[campo]
                    .astype(str)
                    .str.contains(
                        busca,
                        case=False,
                        na=False
                    )
                )

        catalogo = catalogo[mascara]

    st.caption(
        f"{formatar_numero(len(catalogo))} produtos encontrados"
    )

    pagina_atual = st.number_input(
        "Página",
        min_value=1,
        max_value=max(
            1,
            (len(catalogo) + 19) // 20
        ),
        value=1,
        step=1
    )

    inicio = (int(pagina_atual) - 1) * 20

    exibidos = catalogo.iloc[
        inicio:inicio + 20
    ]

    for inicio_linha in range(
        0,
        len(exibidos),
        4
    ):

        colunas = st.columns(4)

        for coluna, (_, produto) in zip(
            colunas,
            exibidos.iloc[
                inicio_linha:inicio_linha + 4
            ].iterrows()
        ):

            with coluna:

                st.container(border=True)

                st.markdown(
                    f"**{produto.get('product_name', 'Produto')}**"
                )

                st.caption(
                    produto.get("category", "")
                )

                st.caption(
                    produto.get("brand", "")
                )

                st.markdown(
                    f"### {formatar_moeda(produto.get('price'))}"
                )

    if exibidos.empty:

        st.info(
            "Nenhum produto encontrado para os filtros selecionados."
        )


# ============================================================
# PÁGINA 5 — INTELIGÊNCIA DO MODELO
# ============================================================

elif pagina == "Inteligência do modelo":

    exibir_secao(
        "Inteligência do modelo",
        "Métricas e informações da versão Item-Item selecionada."
    )

    metricas = metadados.get(
        "metricas",
        metadados.get("metrics", {})
    )

    if not isinstance(metricas, dict):

        metricas = {}

    def localizar_metrica(nomes):

        for nome in nomes:

            if nome in metricas:

                try:

                    return float(metricas[nome])

                except (ValueError, TypeError):

                    pass

        return None

    configuracoes = [
        (
            "Precision@10",
            [
                "precision_at_10",
                "precision@10",
                "precision_10"
            ]
        ),
        (
            "Recall@10",
            [
                "recall_at_10",
                "recall@10",
                "recall_10"
            ]
        ),
        (
            "MAP@10",
            [
                "map_at_10",
                "map@10",
                "map_10"
            ]
        )
    ]

    colunas = st.columns(3)

    for coluna, (nome, alternativas) in zip(
        colunas,
        configuracoes
    ):

        valor = localizar_metrica(
            alternativas
        )

        coluna.metric(
            nome,
            f"{valor:.4f}" if valor is not None else "—"
        )

    st.divider()

    st.subheader(
        "Configuração do modelo"
    )

    st.write(
        "**Modelo:** Similaridade Item-Item"
    )

    st.write(
        f"**Versão:** `{MODEL.name}`"
    )

    st.write(
        "**Avaliação:** métricas offline"
    )

    if metadados:

        with st.expander(
            "Visualizar metadados completos"
        ):

            st.json(metadados)

    else:

        st.info(
            "Arquivo de metadados não encontrado."
        )


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    "RecoCommerce Brasil | Projeto demonstrativo de "
    "sistemas de recomendação e práticas de MLOps."
)