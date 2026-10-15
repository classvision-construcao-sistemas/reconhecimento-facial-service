from app.model.aluno import Aluno
from app.repository.aluno_repository import AlunoRepository, escapar_string
from tests.conftest import FakeMilvusClient


class ClienteEspiao:
    """Regista os argumentos recebidos, para inspecionar os filtros gerados."""

    def __init__(self):
        self.ultima = None

    def query(self, **kwargs):
        self.ultima = kwargs
        return []

    def delete(self, **kwargs):
        self.ultima = kwargs
        return {"delete_count": 0}


def test_escapar_string():
    assert escapar_string('a"b') == 'a\\"b'
    assert escapar_string("a\\b") == "a\\\\b"


def test_filtro_matricula_escapado_contra_injecao():
    espiao = ClienteEspiao()
    repo = AlunoRepository(client=espiao)

    repo.buscar_por_matricula('1" or matricula != "')

    assert espiao.ultima["filter"] == 'matricula == "1\\" or matricula != \\""'


def test_buscar_por_nome_escapa_aspas_e_curingas():
    espiao = ClienteEspiao()
    repo = AlunoRepository(client=espiao)

    repo.buscar_por_nome('x" or nome like "%')

    filtro = espiao.ultima["filter"]
    assert filtro == 'nome like "%x\\" or nome like \\"\\%%"'
    assert "embedding" not in espiao.ultima["output_fields"]
    assert espiao.ultima["limit"] == 100


def test_listar_sem_embedding_e_com_limite_maximo():
    espiao = ClienteEspiao()
    repo = AlunoRepository(client=espiao)

    repo.listar(limite=10_000)

    assert "embedding" not in espiao.ultima["output_fields"]
    assert espiao.ultima["limit"] == 1000


def test_excluir_usa_filtro_por_matricula():
    espiao = ClienteEspiao()
    AlunoRepository(client=espiao).excluir("2024001")
    assert espiao.ultima["filter"] == 'matricula == "2024001"'


def test_atualizar_nao_duplica():
    cliente = FakeMilvusClient()
    repo = AlunoRepository(client=cliente)
    repo.inserir(Aluno(matricula="1", embedding=[0.1] * 512, nome="Antigo"))

    repo.atualizar(Aluno(matricula="1", embedding=[0.2] * 512, nome="Novo"))

    assert len(cliente.registos) == 1
    assert cliente.registos[0]["nome"] == "Novo"


def test_cliente_milvus_nao_e_criado_no_import():
    assert AlunoRepository()._client is None
