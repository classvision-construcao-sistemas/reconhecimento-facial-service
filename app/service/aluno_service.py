from app.config.settings import TAMANHO_MAXIMO_CAMPOS
from app.model.aluno import Aluno
from app.service.exceptions import AlunoJaCadastradoError, AlunoNaoEncontradoError


CAMPOS_OPCIONAIS = ("nome", "telefone", "responsavel", "status")


class AlunoService:
    """
    Camada de domínio. Orquestra o motor de IA (extração de embeddings) e a
    persistência vetorial, mantendo o controller limitado ao protocolo HTTP.
    """

    def __init__(self, repository, face_extraction_service):

        self.repository = repository
        self.face_extraction_service = face_extraction_service

    @staticmethod
    def validar_dados(matricula, dados=None):
        """
        Normaliza e valida a matrícula (obrigatória) e os campos opcionais
        contra os tamanhos máximos do schema da coleção. Lança ValueError.
        """
        matricula = (str(matricula).strip() if matricula is not None else "")
        if not matricula:
            raise ValueError("O campo 'matricula' é obrigatório.")

        valores = {"matricula": matricula}
        for campo in CAMPOS_OPCIONAIS:
            valor = (dados or {}).get(campo)
            valores[campo] = str(valor).strip() if valor is not None else ""

        for campo, valor in valores.items():
            maximo = TAMANHO_MAXIMO_CAMPOS[campo]
            if len(valor) > maximo:
                raise ValueError(
                    f"O campo '{campo}' excede o tamanho máximo de {maximo} caracteres."
                )

        return valores

    def cadastrar(self, matricula, img, dados=None):
        """
        Valida os dados, garante que a matrícula ainda não existe, extrai o
        embedding do rosto principal (DeepFace/ArcFace, confiança > 0.6) e
        persiste o vetor de 512 dimensões com os dados do aluno.
        """

        valores = self.validar_dados(matricula, dados)

        # Verifica a duplicidade antes de gastar CPU com a rede neural
        if self.repository.existe_matricula(valores["matricula"]):
            raise AlunoJaCadastradoError(
                f"Já existe um aluno cadastrado com a matrícula {valores['matricula']}."
            )

        embedding = self.face_extraction_service.process_single_face(img)

        aluno = Aluno(embedding=embedding, **valores)
        self.repository.inserir(aluno)

        return valores

    def buscar(self, matricula):
        aluno = self.repository.buscar_por_matricula(str(matricula).strip())
        if aluno is None:
            raise AlunoNaoEncontradoError(
                f"Aluno com a matrícula {matricula} não encontrado."
            )
        return {campo: aluno.get(campo, "") for campo in ("matricula",) + CAMPOS_OPCIONAIS}

    def excluir(self, matricula):
        removidos = self.repository.excluir(str(matricula).strip())
        if not removidos:
            raise AlunoNaoEncontradoError(
                f"Aluno com a matrícula {matricula} não encontrado."
            )
        return removidos

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
