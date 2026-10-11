# pyrefly: ignore [missing-import]
from flask import Blueprint, request, jsonify
import logging

from app.repository.aluno_repository import AlunoRepository
from app.service.aluno_service import AlunoService
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


@aluno_bp.route('/v1/cadastrar-aluno', methods=['POST'])
def cadastrar_aluno():
    """
    Endpoint POST /api/v1/cadastrar-aluno
    Recebe o student_id e uma imagem via multipart/form-data.
    Extrai o vetor facial e cadastra o aluno na base vetorial.
    """
    try:
        student_id = request.form.get('student_id')
        if not student_id:
            return jsonify({"error": "O campo 'student_id' é obrigatório no form-data."}), 400

        if 'image' not in request.files:
            return jsonify({"error": "O ficheiro de imagem 'image' é obrigatório."}), 400

        file = request.files['image']
        img = image_processor.process_upload(file)

        # Extrai o embedding via DeepFace (ArcFace) e persiste os 512 floats no Milvus
        aluno_service.cadastrar(student_id, img)

        return jsonify({
            "message": f"Aluno {student_id} cadastrado com sucesso.",
            "student_id": student_id
        }), 201

    except ValueError as ve:
        logging.warning(f"Erro de validação ao cadastrar aluno: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Erro interno ao cadastrar aluno: {str(e)}")
        return jsonify({"error": "Erro interno no processamento do ficheiro ou base de dados."}), 500


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
        logging.warning(f"Erro de validação ao auditar sala: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Erro interno ao auditar sala: {str(e)}")
        return jsonify({"error": "Erro interno no processamento do ficheiro ou base de dados."}), 500
