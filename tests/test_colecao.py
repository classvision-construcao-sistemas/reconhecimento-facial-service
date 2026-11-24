import pytest

from app.config import settings
from app.repository.colecao import criar_schema, garantir_colecao


class ClienteFalso:
    def __init__(self, existentes=()):
        self.existentes = set(existentes)
        self.criadas = []

    def has_collection(self, nome):
        return nome in self.existentes

    def prepare_index_params(self):
        class Indices:
            def __init__(self):
                self.itens = []

            def add_index(self, **kwargs):
                self.itens.append(kwargs)

        return Indices()

    def create_collection(self, collection_name, schema, index_params):
        self.criadas.append((collection_name, schema, index_params))
        self.existentes.add(collection_name)


def test_cria_a_colecao_quando_nao_existe():
    cliente = ClienteFalso()
    assert garantir_colecao(cliente, "Aluno") is True
    nome, schema, indices = cliente.criadas[0]
    assert nome == "Aluno"
    assert {campo.name for campo in schema.fields} == {
        "id", "matricula", "nome", "telefone", "embedding", "responsavel", "status"
    }
    assert {"field_name": "embedding", "index_type": "AUTOINDEX", "metric_type": "COSINE"} in indices.itens


def test_nao_recria_colecao_existente():
    cliente = ClienteFalso(existentes={"Aluno"})
    assert garantir_colecao(cliente, "Aluno") is False
    assert cliente.criadas == []


def test_embedding_tem_512_dimensoes():
    embedding = next(campo for campo in criar_schema().fields if campo.name == "embedding")
    assert embedding.params["dim"] == 512


@pytest.mark.parametrize("uri, exige", [
    ("https://in03-abc.serverless.gcp-us-west1.cloud.zilliz.com", True),
    ("http://milvus:19530", False),
    ("./milvus.db", False),
])
def test_so_o_zilliz_cloud_exige_token(uri, exige):
    assert settings.requer_token(uri) is exige


def test_milvus_local_dispensa_token(monkeypatch):
    monkeypatch.setattr(settings, "ZILLIZ_URI", "http://milvus:19530")
    monkeypatch.setattr(settings, "ZILLIZ_TOKEN", "")
    settings.validar_credenciais()


def test_zilliz_sem_token_falha_cedo(monkeypatch):
    monkeypatch.setattr(settings, "ZILLIZ_URI", "https://x.cloud.zilliz.com")
    monkeypatch.setattr(settings, "ZILLIZ_TOKEN", "")
    with pytest.raises(RuntimeError, match="ZILLIZ_TOKEN"):
        settings.validar_credenciais()


def test_sem_uri_falha_cedo(monkeypatch):
    monkeypatch.setattr(settings, "ZILLIZ_URI", None)
    with pytest.raises(RuntimeError, match="ZILLIZ_URI"):
        settings.validar_credenciais()
