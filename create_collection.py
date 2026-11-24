# pyrefly: ignore [missing-import]
from pymilvus import MilvusClient

from app.config.settings import (
    validar_credenciais,
    ZILLIZ_URI,
    ZILLIZ_TOKEN,
    ZILLIZ_COLLECTION
)
from app.repository.colecao import garantir_colecao


# =========================================================
# CONEXÃO COM ZILLIZ
# =========================================================

validar_credenciais()

client = MilvusClient(
    uri=ZILLIZ_URI,
    token=ZILLIZ_TOKEN
)


# =========================================================
# CRIAR A COLEÇÃO (schema e índices em app/repository/colecao.py)
# =========================================================

if garantir_colecao(client, ZILLIZ_COLLECTION):
    print(f"Coleção '{ZILLIZ_COLLECTION}' criada com sucesso!")
else:
    print(f"A coleção '{ZILLIZ_COLLECTION}' já existe.")


# =========================================================
# MOSTRAR ESTRUTURA DA COLEÇÃO
# =========================================================

descricao = client.describe_collection(
    collection_name=ZILLIZ_COLLECTION
)

print("\nEstrutura da coleção:")
print(descricao)


# =========================================================
# MOSTRAR ÍNDICES
# =========================================================

indices = client.list_indexes(
    collection_name=ZILLIZ_COLLECTION
)

print("\nÍndices:")
print(indices)