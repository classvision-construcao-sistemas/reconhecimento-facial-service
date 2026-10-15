# pyrefly: ignore [missing-import]
from pymilvus import MilvusClient, DataType

from app.config.settings import (
    validar_credenciais,
    ZILLIZ_URI,
    ZILLIZ_TOKEN,
    ZILLIZ_COLLECTION
)


# =========================================================
# CONEXÃO COM ZILLIZ
# =========================================================

validar_credenciais()

client = MilvusClient(
    uri=ZILLIZ_URI,
    token=ZILLIZ_TOKEN
)


# =========================================================
# CRIAR SCHEMA
# =========================================================

schema = MilvusClient.create_schema(
    auto_id=True,
    enable_dynamic_field=False
)


# =========================================================
# CAMPOS
# =========================================================

schema.add_field(
    field_name="id",
    datatype=DataType.INT64,
    is_primary=True,
    auto_id=True
)

schema.add_field(
    field_name="matricula",
    datatype=DataType.VARCHAR,
    max_length=20
)

schema.add_field(
    field_name="nome",
    datatype=DataType.VARCHAR,
    max_length=150
)

schema.add_field(
    field_name="telefone",
    datatype=DataType.VARCHAR,
    max_length=20
)

schema.add_field(
    field_name="embedding",
    datatype=DataType.FLOAT_VECTOR,
    dim=512
)

schema.add_field(
    field_name="responsavel",
    datatype=DataType.VARCHAR,
    max_length=150
)

schema.add_field(
    field_name="status",
    datatype=DataType.VARCHAR,
    max_length=20
)


# =========================================================
# CRIAR ÍNDICES
# =========================================================

index_params = client.prepare_index_params()

# Índice da chave primária
index_params.add_index(
    field_name="id",
    index_type="AUTOINDEX"
)

# Índice do embedding facial
index_params.add_index(
    field_name="embedding",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)


# =========================================================
# VERIFICAR SE A COLEÇÃO JÁ EXISTE
# =========================================================

if client.has_collection(ZILLIZ_COLLECTION):

    print(f"A coleção '{ZILLIZ_COLLECTION}' já existe.")

else:

    client.create_collection(
        collection_name=ZILLIZ_COLLECTION,
        schema=schema,
        index_params=index_params
    )

    print(f"Coleção '{ZILLIZ_COLLECTION}' criada com sucesso!")


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