from app.model.aluno import Aluno


class AlunoService:
    """
    Camada de domínio. Orquestra o motor de IA (extração de embeddings) e a
    persistência vetorial, mantendo o controller limitado ao protocolo HTTP.
    """

    def __init__(self, repository, face_extraction_service):

        self.repository = repository
        self.face_extraction_service = face_extraction_service

    def cadastrar(self, matricula, img):
        """
        Extrai o embedding do rosto principal (DeepFace/ArcFace, confiança > 0.6)
        e persiste o vetor de 512 dimensões associado à matrícula.
        """

        embedding = self.face_extraction_service.process_single_face(img)

        # Campos biográficos vão vazios: só biometria + matrícula são guardados
        aluno = Aluno(
            matricula=str(matricula),
            embedding=embedding,
            nome="",
            telefone="",
            responsavel="",
            status=""
        )

        return self.repository.inserir(aluno)

    def auditar_sala(self, img):
        """
        Deteta múltiplos rostos na foto da turma, compara os embeddings com a
        base vetorial e devolve o payload de auditoria já no formato consumido
        pelo sistema principal em Java.
        """

        # Deteta rostos descartando confianças <= 0.6
        faces = self.face_extraction_service.process_multiple_faces(img)

        if not faces:
            return {
                "message": "Auditoria concluída. Nenhum rosto válido detetado na imagem.",
                "recognized_students": [],
                "unrecognized_count": 0,
                "total_faces_detected": 0
            }

        embeddings = [face["embedding"] for face in faces]

        # Executa a busca de similaridade (distância cosseno) no Milvus
        search_results = self.repository.buscar_faces(embeddings, threshold=0.5)

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

        return {
            "message": "Auditoria concluída com sucesso.",
            "recognized_students": recognized,
            "unrecognized_count": unrecognized_count,
            "total_faces_detected": len(faces)
        }
