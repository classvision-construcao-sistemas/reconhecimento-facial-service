class Aluno:
    """
    Entidade persistida na coleção vetorial do Zilliz Cloud.

    A matrícula e o embedding facial são obrigatórios. Os campos biográficos
    (nome, telefone, responsável, status) são opcionais no cadastro; como são
    obrigatórios no schema da coleção, recebem string vazia quando omitidos.
    """

    def __init__(
        self,
        matricula,
        embedding,
        nome="",
        telefone="",
        responsavel="",
        status=""
    ):
        self.matricula = matricula
        self.embedding = embedding
        self.nome = nome
        self.telefone = telefone
        self.responsavel = responsavel
        self.status = status
