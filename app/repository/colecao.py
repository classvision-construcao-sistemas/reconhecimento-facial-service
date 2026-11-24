# pyrefly: ignore [missing-import]
from pymilvus import DataType, MilvusClient

from app.config.settings import TAMANHO_MAXIMO_CAMPOS


DIMENSAO_EMBEDDING = 512


def criar_schema():
    """Schema da coleção de alunos (o mesmo usado no Zilliz Cloud)."""
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="matricula", datatype=DataType.VARCHAR, max_length=TAMANHO_MAXIMO_CAMPOS["matricula"])
    schema.add_field(field_name="nome", datatype=DataType.VARCHAR, max_length=TAMANHO_MAXIMO_CAMPOS["nome"])
    schema.add_field(field_name="telefone", datatype=DataType.VARCHAR, max_length=TAMANHO_MAXIMO_CAMPOS["telefone"])
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=DIMENSAO_EMBEDDING)
    schema.add_field(field_name="responsavel", datatype=DataType.VARCHAR, max_length=TAMANHO_MAXIMO_CAMPOS["responsavel"])
    schema.add_field(field_name="status", datatype=DataType.VARCHAR, max_length=TAMANHO_MAXIMO_CAMPOS["status"])
    return schema


def criar_indices(client):
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="id", index_type="AUTOINDEX")
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
    return index_params


def garantir_colecao(client, nome):
    """Cria a coleção se ainda não existir. Devolve True quando a criou."""
    if client.has_collection(nome):
        return False
    client.create_collection(collection_name=nome, schema=criar_schema(), index_params=criar_indices(client))
    return True
