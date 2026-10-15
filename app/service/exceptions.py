class AlunoJaCadastradoError(Exception):
    """A matrícula informada já possui biometria cadastrada (HTTP 409)."""


class AlunoNaoEncontradoError(Exception):
    """Não existe aluno cadastrado com a matrícula informada (HTTP 404)."""
