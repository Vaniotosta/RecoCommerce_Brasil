
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, save_npz

# ============================================================
# 1. CAMINHOS
# ============================================================

RAIZ = Path(__file__).resolve().parents[2]

CAMINHO_DADOS = (
    RAIZ / "data" / "processed" / "interacoes_ponderadas.csv"
)

PASTA_MODELOS = RAIZ / "data" / "processed" / "matrizes"
PASTA_GRAFICOS = RAIZ / "reports" / "figures"
PASTA_TABELAS = RAIZ / "reports" / "tabelas"

for pasta in [PASTA_MODELOS, PASTA_GRAFICOS, PASTA_TABELAS]:
    pasta.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LEITURA DOS DADOS
# ============================================================

dados = pd.read_csv(CAMINHO_DADOS)

print("Interações ponderadas:", len(dados))

# ============================================================
# 3. MAPEAMENTO DOS IDENTIFICADORES
# ============================================================

clientes = np.sort(dados["customer_id"].unique())
produtos = np.sort(dados["product_id"].unique())

mapa_clientes = {
    cliente: indice
    for indice, cliente in enumerate(clientes)
}

mapa_produtos = {
    produto: indice
    for indice, produto in enumerate(produtos)
}

linhas = dados["customer_id"].map(mapa_clientes).to_numpy()
colunas = dados["product_id"].map(mapa_produtos).to_numpy()
valores = dados["pontuacao"].to_numpy(dtype=np.float32)

# ============================================================
# 4. CONSTRUÇÃO DA MATRIZ ESPARSA
# ============================================================

matriz_esparsa = csr_matrix(
    (valores, (linhas, colunas)),
    shape=(len(clientes), len(produtos)),
    dtype=np.float32
)

# ============================================================
# 5. INDICADORES
# ============================================================

total_posicoes = (
    matriz_esparsa.shape[0] * matriz_esparsa.shape[1]
)

posicoes_preenchidas = matriz_esparsa.nnz

densidade = (
    posicoes_preenchidas / total_posicoes
) * 100

esparsidade = 100 - densidade

memoria_esparsa = (
    matriz_esparsa.data.nbytes
    + matriz_esparsa.indices.nbytes
    + matriz_esparsa.indptr.nbytes
)

memoria_densa = (
    matriz_esparsa.shape[0]
    * matriz_esparsa.shape[1]
    * np.dtype(np.float32).itemsize
)

print("\n===== MATRIZ ESPARSA =====")
print("Dimensões:", matriz_esparsa.shape)
print("Posições preenchidas:", posicoes_preenchidas)
print(f"Densidade: {densidade:.4f}%")
print(f"Esparsidade: {esparsidade:.4f}%")
print(f"Memória densa: {memoria_densa / 1024**2:.2f} MB")
print(f"Memória esparsa: {memoria_esparsa / 1024**2:.2f} MB")

# ============================================================
# 6. SALVAMENTO DA MATRIZ E DOS IDENTIFICADORES
# ============================================================

save_npz(
    PASTA_MODELOS / "matriz_cliente_produto.npz",
    matriz_esparsa
)

np.save(
    PASTA_MODELOS / "clientes_ids.npy",
    clientes
)

np.save(
    PASTA_MODELOS / "produtos_ids.npy",
    produtos
)

# ============================================================
# 7. GRÁFICO DE DENSIDADE E ESPARSIDADE
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    ["Densidade", "Esparsidade"],
    [densidade, esparsidade]
)

plt.title("Densidade e Esparsidade da Matriz Cliente-Produto")
plt.ylabel("Percentual (%)")
plt.ylim(0, 100)

for i, valor in enumerate([densidade, esparsidade]):
    plt.text(i, valor + 1, f"{valor:.2f}%", ha="center")

plt.tight_layout()

plt.savefig(
    PASTA_GRAFICOS / "densidade_esparsidade_matriz.png",
    dpi=300
)

plt.close()

# ============================================================
# 8. RELATÓRIO DE MEMÓRIA
# ============================================================

relatorio = pd.DataFrame({
    "indicador": [
        "clientes",
        "produtos",
        "posicoes_totais",
        "posicoes_preenchidas",
        "densidade_pct",
        "esparsidade_pct",
        "memoria_densa_mb",
        "memoria_esparsa_mb"
    ],
    "valor": [
        len(clientes),
        len(produtos),
        total_posicoes,
        posicoes_preenchidas,
        densidade,
        esparsidade,
        memoria_densa / 1024**2,
        memoria_esparsa / 1024**2
    ]
})

relatorio.to_csv(
    PASTA_TABELAS / "relatorio_matriz_esparsa.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivos gerados:")
print("- matriz_cliente_produto.npz")
print("- clientes_ids.npy")
print("- produtos_ids.npy")
print("- densidade_esparsidade_matriz.png")
print("- relatorio_matriz_esparsa.csv")

print("\nProcessamento concluído!")
