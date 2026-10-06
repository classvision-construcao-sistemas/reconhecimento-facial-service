from flask import Blueprint, request, jsonify

from app.repository.aluno_repository import (
    AlunoRepository
)

from app.service.aluno_service import (
    AlunoService
)


aluno_bp = Blueprint(
    "aluno",
    __name__
)


repository = AlunoRepository()

service = AlunoService(
    repository
)


@aluno_bp.route(
    "/alunos",
    methods=["POST"]
)
def cadastrar():

    dados = request.get_json()

    resultado = service.cadastrar(
        nome=dados["nome"],
        curso=dados["curso"],
        periodo=dados["periodo"],
        embedding=dados["embedding"]
    )

    return jsonify(resultado), 201