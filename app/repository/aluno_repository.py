# pyrefly: ignore [missing-import]
from pymilvus import MilvusClient

from app.config import settings


# Campos devolvidos nas leituras. O embedding fica de fora de propósito:
# são 512 floats por aluno e nenhum consumidor precisa deles numa listagem.
CAMPOS_SAIDA = ["matricula", "nome", "telefone", "responsavel", "status"]


def escapar_string(valor) -> str:
    """
    Escapa um valor para uso seguro dentro de uma string entre aspas duplas
    numa expressão de filtro do Milvus (evita injeção de filtro).
    """
    return str(valor).replace("\\", "\\\\").replace('"', '\\"')


class AlunoRepository:

    def __init__(self, client=None):
        # O cliente é criado de forma preguiçosa (na primeira utilização), para
        # que importar a aplicação não exija credenciais nem acesso à rede.
        self._client = client

    @property
    def client(self):
        if self._client is None:
            settings.validar_credenciais()
            self._client = MilvusClient(
                uri=settings.ZILLIZ_URI,
                token=settings.ZILLIZ_TOKEN
            )
        return self._client

    @property
    def collection(self):
        return settings.ZILLIZ_COLLECTION

    @staticmethod
    def _filtro_matricula(matricula) -> str:
        return f'matricula == "{escapar_string(matricula)}"'

    @staticmethod
    def _limite(limite) -> int:
        try:
            limite = int(limite)
        except (TypeError, ValueError):
            limite = settings.LIMITE_LISTAGEM_PADRAO
        return max(1, min(limite, settings.LIMITE_LISTAGEM_MAXIMO))

    @staticmethod
    def _dados(aluno) -> dict:
        return {
            # Converte para string por segurança, caso o Zilliz espere um VARCHAR
            "matricula": str(aluno.matricula),
            "nome": aluno.nome,
            "telefone": aluno.telefone,
            "embedding": aluno.embedding,
            "responsavel": aluno.responsavel,
            "status": aluno.status
        }

    # CREATE
    def inserir(self, aluno):
        return self.client.insert(
            collection_name=self.collection,
            data=[self._dados(aluno)]
        )

    # READ - busca em lote dos rostos detetados numa sala
    def buscar_faces(self, embeddings, threshold=0.5, limite=1):
        """Recebe os vetores, aplica o limiar de confiança e formata a resposta."""
        self.client.load_collection(collection_name=self.collection)

        # O Milvus devolve uma lista de resultados para cada rosto procurado
        resultados_brutos = self.client.search(
            collection_name=self.collection,
            data=embeddings,
            anns_field="embedding",
            limit=limite,
            output_fields=["matricula"]
        )

        search_results = []

        # Formata a resposta para encaixar perfeitamente no AlunoService
        for hits in resultados_brutos:
            if hits and len(hits) > 0:
                melhor_resultado = hits[0]
                score = melhor_resultado.get("distance", 0.0)

                # Só considera válido se a distância cosseno atingir o threshold
                if score >= threshold:
                    search_results.append({
                        "student_id": melhor_resultado["entity"]["matricula"],
                        "score": score
                    })
                else:
                    search_results.append(None) # Rosto detetado, mas não reconhecido
            else:
                search_results.append(None)

        return search_results

    # READ - buscar por embedding
    def buscar_por_embedding(self, embedding, limite=1):

        # Carrega a coleção na memória
        self.client.load_collection(collection_name=self.collection)

        # Realiza a busca por similaridade
        return self.client.search(
            collection_name=self.collection,
            data=[embedding],
            anns_field="embedding",
            limit=limite,
            output_fields=CAMPOS_SAIDA,
        )

    # READ - buscar por matrícula (a chave primária é o "id" auto-gerado,
    # por isso a matrícula tem de ser procurada por filtro)
    def buscar_por_matricula(self, matricula):

        resultados = self.client.query(
            collection_name=self.collection,
            filter=self._filtro_matricula(matricula),
            output_fields=CAMPOS_SAIDA,
            limit=1
        )

        return resultados[0] if resultados else None

    def existe_matricula(self, matricula) -> bool:
        return self.buscar_por_matricula(matricula) is not None

    # READ - listar todos
    def listar(self, limite=settings.LIMITE_LISTAGEM_PADRAO):

        return self.client.query(
            collection_name=self.collection,
            filter="matricula != ''",
            output_fields=CAMPOS_SAIDA,
            limit=self._limite(limite)
        )

    # READ - buscar por nome
    def buscar_por_nome(self, nome, limite=settings.LIMITE_LISTAGEM_PADRAO):

        # Escapa aspas/barras e os curingas do LIKE, para que o texto do
        # utilizador seja tratado como literal e não como parte do filtro.
        termo = escapar_string(nome).replace("%", "\\%").replace("_", "\\_")

        return self.client.query(
            collection_name=self.collection,
            filter=f'nome like "%{termo}%"',
            output_fields=CAMPOS_SAIDA,
            limit=self._limite(limite)
        )

    # UPDATE - remove o registo da matrícula e insere o novo, evitando que o
    # upsert sem "id" (auto_id) crie um registo duplicado
    def atualizar(self, aluno):

        self.excluir(aluno.matricula)
        return self.inserir(aluno)

    # DELETE - devolve o número de registos removidos
    def excluir(self, matricula) -> int:

        resultado = self.client.delete(
            collection_name=self.collection,
            filter=self._filtro_matricula(matricula)
        )

        if isinstance(resultado, dict):
            return int(resultado.get("delete_count", 0))
        return int(getattr(resultado, "delete_count", 0) or 0)
