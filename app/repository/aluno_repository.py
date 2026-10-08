# pyrefly: ignore [missing-import]
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

    # CREATE
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

    # READ - buscar por embedding
    def buscar_por_embedding(self, embedding, limite=1):

        # Carrega a coleção na memória
        self.client.load_collection(
            collection_name=ZILLIZ_COLLECTION
        )

        # Realiza a busca por similaridade
        resultados = self.client.search(
            collection_name=ZILLIZ_COLLECTION,
            data=[embedding],
            anns_field="embedding",
            limit=limite,
            output_fields=[
                "matricula",
                "nome",
                "telefone",
                "responsavel",
                "status"
            ],
        )

        return resultados

    # READ - buscar por matrícula
    def buscar_por_matricula(self, matricula):

        return self.client.get(
            collection_name=ZILLIZ_COLLECTION,
            ids=[matricula]
        )

    # READ - listar todos
    def listar(self):

        return self.client.query(
            collection_name=ZILLIZ_COLLECTION,
            filter="matricula != ''",
            output_fields=[
                "matricula",
                "nome",
                "telefone",
                "embedding",
                "responsavel",
                "status"
            ]
        )

    # READ - buscar por nome
    def buscar_por_nome(self, nome):

        return self.client.query(
            collection_name=ZILLIZ_COLLECTION,
            filter=f'nome like "%{nome}%"',
            output_fields=[
                "matricula",
                "nome",
                "telefone",
                "embedding",
                "responsavel",
                "status"
            ]
        )

    # UPDATE
    def atualizar(self, aluno):

        dados = {
            "matricula": aluno.matricula,
            "nome": aluno.nome,
            "telefone": aluno.telefone,
            "embedding": aluno.embedding,
            "responsavel": aluno.responsavel,
            "status": aluno.status
        }

        return self.client.upsert(
            collection_name=ZILLIZ_COLLECTION,
            data=[dados]
        )

    # DELETE
    def excluir(self, matricula):

        return self.client.delete(
            collection_name=ZILLIZ_COLLECTION,
            ids=[matricula]
        )