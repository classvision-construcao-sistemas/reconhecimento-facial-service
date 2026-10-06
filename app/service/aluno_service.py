from app.model.aluno import Aluno


class AlunoService:

    def __init__(self, repository):

        self.repository = repository

    def cadastrar(
        self,
        nome,
        curso,
        periodo,
        embedding
    ):

        aluno = Aluno(
            nome=nome,
            curso=curso,
            periodo=periodo,
            embedding=embedding
        )

        return self.repository.inserir(aluno)