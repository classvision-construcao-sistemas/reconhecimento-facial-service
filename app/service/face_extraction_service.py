# pyrefly: ignore [missing-import]
from deepface import DeepFace
# pyrefly: ignore [missing-import]
import numpy as np
from typing import List, Dict, Any
# pyrefly: ignore [missing-import]
import cv2
import os
import uuid


class FaceExtractionService:
    """
    O motor de Inteligência Artificial para deteção, alinhamento, recorte e extração de embeddings faciais.
    Utiliza RetinaFace para deteção e ArcFace para obter o vetor de 512 dimensões.
    """
    def __init__(self):
        self.detector_backend = "mtcnn"
        self.model_name = "ArcFace"
        self.confidence_threshold = 0.6

    def process_single_face(self, img: np.ndarray) -> List[float]:
        """
        Processa uma foto única para registo. Extrai o embedding do rosto principal.
        Garante que deteções com confiança <= 0.6 são descartadas.
        """
        temp_filename = f"temp_single_face_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(temp_filename, img)

        try:
            results = DeepFace.represent(
                img_path=temp_filename,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True
            )

            if not results:
                raise ValueError("Nenhum rosto detetado na imagem.")

            # Para cadastro, selecionamos o rosto com maior confiança na imagem
            best_face = max(results, key=lambda x: x.get("face_confidence", 0))

            if best_face.get("face_confidence", 0) <= self.confidence_threshold:
                raise ValueError(f"Confiança do rosto detetado ({best_face.get('face_confidence')}) é inferior ou igual ao limite exigido ({self.confidence_threshold}). Falso positivo descartado.")

            return best_face["embedding"]

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def process_multiple_faces(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Processa uma foto densa de uma sala de aula.
        Descarta deteções de rosto com confiança <= 0.6.
        Retorna uma lista contendo os embeddings e a bounding box.
        """
        temp_filename = f"temp_multi_face_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(temp_filename, img)

        try:
            results = DeepFace.represent(
                img_path=temp_filename,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False # Permite que não haja erros se nenhum rosto for detetado
            )
        except Exception as e:
            print(f"\n[ERRO CRÍTICO IA] Falha na extração DeepFace: {str(e)}\n")
            return []
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        valid_faces = []
        for face in results:
            # Qualquer deteção com confiança igual ou inferior a 0.6 é descartada
            if face.get("face_confidence", 0) > self.confidence_threshold:
                valid_faces.append({
                    "embedding": face["embedding"],
                    "facial_area": face["facial_area"],
                    "confidence": face.get("face_confidence")
                })

        return valid_faces
