class Aluno:

    def __init__(
        self,
        matricula,
        nome,
        telefone,
        embedding,
        responsavel,
        status
    ):
        self.matricula = matricula
        self.nome = nome
        self.telefone = telefone
        self.embedding = embedding
        self.responsavel = responsavel
        self.status = status