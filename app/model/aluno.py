class Aluno:
    """
    Entidade persistida na coleção vetorial do Zilliz Cloud.

    O motor biométrico só conhece a matrícula e o embedding facial. Os campos
    biográficos (nome, telefone, responsável, status) são obrigatórios no schema
    da coleção, por isso recebem string vazia por omissão — é esse o "bypass"
    que mantém a inserção válida guardando apenas biometria + matrícula.
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
