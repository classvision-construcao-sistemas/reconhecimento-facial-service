from pymilvus import MilvusClient

from app.config.settings import (
    ZILLIZ_URI,
    ZILLIZ_TOKEN,
    ZILLIZ_COLLECTION
)


class AlunoRepository:

    def __init__(self):

        self.client = MilvusClient(
            uri=ZILLIZ_URI,
            token=ZILLIZ_TOKEN
        )

    def inserir(self, aluno):

        dados = {
        "matricula": aluno.matricula,
        "nome": aluno.nome,
        "telefone": aluno.telefone,
        "embedding": aluno.embedding,
        "responsavel": aluno.responsavel,
        "status": aluno.status
        }

        return self.client.insert(
            collection_name=ZILLIZ_COLLECTION,
            data=[dados]
        )