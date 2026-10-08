# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify
# pyrefly: ignore [missing-import]
from image_processor import ImageProcessor
# pyrefly: ignore [missing-import]
from face_extraction_service import FaceExtractionService
# pyrefly: ignore [missing-import]
from milvus_repository import MilvusRepository
import logging

class AlunoController:
    """
    Porta de entrada do Flask. Fica responsável por instanciar a aplicação, compor 
    as dependências (ImageProcessor, FaceExtractionService, MilvusRepository) 
    e expor os endpoints principais (cadastro e auditoria).
    """
    def __init__(self):
        self.app = Flask(__name__)
        self.image_processor = ImageProcessor()
        self.face_extraction_service = FaceExtractionService()
        self.milvus_repository = MilvusRepository()
        self._setup_routes()

    def _setup_routes(self):
        """Mapeia as rotas HTTP para os métodos da controladora."""
        self.app.route('/api/v1/cadastrar-aluno', methods=['POST'])(self.cadastrar_aluno)
        self.app.route('/api/v1/auditar-sala', methods=['POST'])(self.auditar_sala)

    def cadastrar_aluno(self):
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
            img = self.image_processor.process_upload(file)
            
            # Obtém o embedding via DeepFace (ArcFace) validando no mínimo 0.6 de confiança (RetinaFace)
            embedding = self.face_extraction_service.process_single_face(img)
            
            # Persiste o vetor de 512 dimensões no Milvus
            self.milvus_repository.insert_student(student_id, embedding)
            
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

    def auditar_sala(self):
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
            img = self.image_processor.process_upload(file)
            
            # Deteta rostos descartando confianças <= 0.6
            faces = self.face_extraction_service.process_multiple_faces(img)
            
            if not faces:
                return jsonify({
                    "message": "Auditoria concluída. Nenhum rosto válido detetado na imagem.",
                    "recognized_students": [],
                    "unrecognized_count": 0,
                    "total_faces_detected": 0
                }), 200

            embeddings = [face["embedding"] for face in faces]
            
            # Executa a busca de similaridade (distância cosseno) no Milvus
            search_results = self.milvus_repository.search_faces(embeddings, threshold=0.5)
            
            recognized = []
            unrecognized_count = 0
            
            for i, match in enumerate(search_results):
                if match:
                    recognized.append({
                        "student_id": match["student_id"],
                        "confidence_score": match["score"],
                        "facial_area": faces[i]["facial_area"]
                    })
                else:
                    unrecognized_count += 1
                    
            return jsonify({
                "message": "Auditoria concluída com sucesso.",
                "recognized_students": recognized,
                "unrecognized_count": unrecognized_count,
                "total_faces_detected": len(faces)
            }), 200

        except ValueError as ve:
            logging.warning(f"Erro de validação ao auditar sala: {str(ve)}")
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            logging.error(f"Erro interno ao auditar sala: {str(e)}")
            return jsonify({"error": "Erro interno no processamento do ficheiro ou base de dados."}), 500

    def get_app(self):
        """Retorna a instância Flask para ser executada ou testada."""
        return self.app