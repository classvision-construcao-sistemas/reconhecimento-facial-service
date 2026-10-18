# pyrefly: ignore [missing-import]
from flask import Blueprint, request, jsonify
import logging

from app.repository.aluno_repository import AlunoRepository
from app.service.aluno_service import AlunoService, CAMPOS_OPCIONAIS
from app.service.exceptions import AlunoJaCadastradoError, AlunoNaoEncontradoError
from app.service.face_extraction_service import FaceExtractionService
from app.util.image_processor import ImageProcessor


# O blueprint é registado em app/__init__.py com url_prefix="/api",
# logo as rotas abaixo ficam expostas como /api/v1/...
aluno_bp = Blueprint("aluno", __name__)


# Composição das dependências (uma única instância por processo)
image_processor = ImageProcessor()
face_extraction_service = FaceExtractionService()
aluno_repository = AlunoRepository()
aluno_service = AlunoService(aluno_repository, face_extraction_service)


@aluno_bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "vector_store": "milvus",
    }), 200


@aluno_bp.route('/v1/cadastrar-aluno', methods=['POST'])
def cadastrar_aluno():
    """
    Endpoint POST /api/v1/cadastrar-aluno
    Recebe via multipart/form-data a matrícula (campo 'student_id' ou
    'matricula'), a imagem ('image') e, opcionalmente, nome, telefone,
    responsavel e status. Extrai o vetor facial e cadastra o aluno.
    """
    try:
        matricula = request.form.get('student_id') or request.form.get('matricula')
        if not matricula or not matricula.strip():
            return jsonify({"error": "O campo 'student_id' (matrícula) é obrigatório no form-data."}), 400

        if 'image' not in request.files:
            return jsonify({"error": "O ficheiro de imagem 'image' é obrigatório."}), 400

        dados = {campo: request.form.get(campo) for campo in CAMPOS_OPCIONAIS}

        # Valida tamanhos antes de descodificar a imagem
        aluno_service.validar_dados(matricula, dados)

        file = request.files['image']
        img = image_processor.process_upload(file)

        # Extrai o embedding via DeepFace (ArcFace) e persiste os 512 floats no Milvus
        aluno = aluno_service.cadastrar(matricula, img, dados)

        return jsonify({
            "message": f"Aluno {aluno['matricula']} cadastrado com sucesso.",
            "student_id": aluno["matricula"],
            "aluno": aluno
        }), 201

    except AlunoJaCadastradoError as e:
        return jsonify({"error": str(e)}), 409
    except ValueError as ve:
        logging.exception(f"Erro de validação ao cadastrar aluno: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Erro interno ao cadastrar aluno: {str(e)}")
        return jsonify({"error": "Erro interno no processamento do ficheiro ou base de dados."}), 500


@aluno_bp.route('/v1/alunos/<matricula>', methods=['GET'])
def buscar_aluno(matricula):
    """Endpoint GET /api/v1/alunos/<matricula> — dados do aluno (sem embedding)."""
    try:
        return jsonify(aluno_service.buscar(matricula)), 200
    except AlunoNaoEncontradoError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.error(f"Erro interno ao buscar aluno: {str(e)}")
        return jsonify({"error": "Erro interno ao consultar a base de dados."}), 500


@aluno_bp.route('/v1/alunos/<matricula>', methods=['DELETE'])
def excluir_aluno(matricula):
    """Endpoint DELETE /api/v1/alunos/<matricula> — remove o aluno e a biometria."""
    try:
        aluno_service.excluir(matricula)
        return jsonify({
            "message": f"Aluno {matricula} excluído com sucesso.",
            "student_id": matricula
        }), 200
    except AlunoNaoEncontradoError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.error(f"Erro interno ao excluir aluno: {str(e)}")
        return jsonify({"error": "Erro interno ao consultar a base de dados."}), 500


@aluno_bp.route('/v1/auditar-sala', methods=['POST'])
def auditar_sala():
    """
    Endpoint POST /api/v1/auditar-sala
    Recebe a foto da sala via multipart/form-data.
    Deteta múltiplos rostos e compara os embeddings com o Milvus.
    Retorna a lista de alunos detetados.
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "O ficheiro de imagem 'image' é obrigatório."}), 400

        file = request.files['image']
        img = image_processor.process_upload(file)

        resultado = aluno_service.auditar_sala(img)

        return jsonify(resultado), 200

    except ValueError as ve:
        logging.exception(f"Erro de validação ao auditar sala: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Erro interno ao auditar sala: {str(e)}")
        return jsonify({"error": "Erro interno no processamento do ficheiro ou base de dados."}), 500
