import logging
import threading
from typing import Any, Dict, List

# pyrefly: ignore [missing-import]
import numpy as np

from app.config import settings
from app.util.image_processor import ImageProcessor
from app.util.model_downloader import garantir_modelo


# Piso interno do SCRFD. O filtro que vale para o negócio é LIMIAR_DETECCAO,
# aplicado depois; este piso só existe para o cadastro conseguir distinguir
# "nenhum rosto" de "rosto com confiança insuficiente", como antes.
_PISO_DETECTOR = 0.3


class FaceExtractionService:
    """
    O motor de Inteligência Artificial para deteção, alinhamento, recorte e
    extração de embeddings faciais:

      1. SCRFD-10G deteta os rostos numa cópia reduzida da foto e devolve
         5 landmarks por rosto (olhos, nariz, cantos da boca);
      2. os landmarks são levados para a resolução original e cada rosto é
         alinhado em 112x112 (norm_crop) a partir da foto original;
      3. o TopoFR R50 converte todos os rostos, num único lote, em vetores de
         512 dimensões normalizados (L2).

    Os modelos só são carregados no primeiro uso, para que importar a
    aplicação (testes, Swagger, health check) não exija os ficheiros .onnx.
    """
    def __init__(self):
        self.confidence_threshold = settings.LIMIAR_DETECCAO
        self._detector = None
        self._reconhecedor = None
        self._entrada_reconhecedor = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Carregamento dos modelos
    # ------------------------------------------------------------------

    def _carregar_modelos(self):
        if self._reconhecedor is not None:
            return
        with self._lock:
            if self._reconhecedor is not None:
                return

            # pyrefly: ignore [missing-import]
            import onnxruntime as ort
            # pyrefly: ignore [missing-import]
            from insightface.model_zoo import get_model

            caminho_scrfd = garantir_modelo(
                settings.MODELS_DIR, settings.SCRFD_ONNX_FICHEIRO,
                settings.SCRFD_ZIP_URL, settings.SCRFD_ONNX_SHA256,
                membro_zip=settings.SCRFD_ONNX_FICHEIRO,
            )
            caminho_topofr = garantir_modelo(
                settings.MODELS_DIR, settings.TOPOFR_ONNX_FICHEIRO,
                settings.TOPOFR_ONNX_URL, settings.TOPOFR_ONNX_SHA256,
            )

            detector = get_model(caminho_scrfd, providers=["CPUExecutionProvider"])
            detector.prepare(
                ctx_id=-1,
                input_size=settings.TAMANHO_ENTRADA_DETECTOR,
                det_thresh=_PISO_DETECTOR,
            )

            reconhecedor = ort.InferenceSession(caminho_topofr, providers=["CPUExecutionProvider"])

            self._detector = detector
            self._entrada_reconhecedor = reconhecedor.get_inputs()[0].name
            self._reconhecedor = reconhecedor
            logging.info("Motor biométrico carregado: SCRFD-10G + TopoFR R50 (Glint360K).")

    # ------------------------------------------------------------------
    # Etapas do pipeline
    # ------------------------------------------------------------------

    def _detectar(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Deteta os rostos na cópia reduzida (o mesmo referencial que o
        facial_area sempre usou) e devolve os landmarks já na escala original.
        """
        img_det = ImageProcessor.redimensionar(img, settings.LARGURA_DETECCAO)
        escala = img.shape[1] / float(img_det.shape[1])
        altura_det, largura_det = img_det.shape[:2]

        bboxes, kpss = self._detector.detect(img_det, max_num=0)
        if bboxes is None or len(bboxes) == 0:
            return []

        rostos = []
        for bbox, kps in zip(bboxes, kpss):
            # float() tira o tipo do numpy, que o JSON não serializa.
            x1, y1, x2, y2, score = (float(v) for v in bbox)
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(largura_det), x2), min(float(altura_det), y2)
            rostos.append({
                "facial_area": {
                    "x": int(round(x1)),
                    "y": int(round(y1)),
                    "w": int(round(x2 - x1)),
                    "h": int(round(y2 - y1)),
                },
                # Mesma caixa em frações (0 a 1) da foto, para o app desenhar
                # sobre a imagem em qualquer tamanho.
                "relative_area": {
                    "x": round(x1 / largura_det, 4),
                    "y": round(y1 / altura_det, 4),
                    "w": round((x2 - x1) / largura_det, 4),
                    "h": round((y2 - y1) / altura_det, 4),
                },
                "confidence": float(score),
                "landmarks": np.asarray(kps, dtype=np.float32) * escala,
            })
        return rostos

    @staticmethod
    def _alinhar(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Leva os 5 landmarks ao molde canónico do ArcFace e recorta 112x112."""
        # pyrefly: ignore [missing-import]
        from insightface.utils import face_align

        return face_align.norm_crop(img, landmarks, image_size=112)

    def _vetorizar(self, img: np.ndarray, rostos: List[Dict[str, Any]]) -> np.ndarray:
        """Alinha cada rosto a partir da foto original e vetoriza todos de uma vez."""
        recortes = [self._alinhar(img, r["landmarks"]) for r in rostos]

        # Mesmo pré-processamento do treino do TopoFR: BGR->RGB, (x/255 - 0.5)/0.5, NCHW
        lote = np.stack(recortes)[:, :, :, ::-1].astype(np.float32)
        lote = (lote / 255.0 - 0.5) / 0.5
        lote = np.ascontiguousarray(lote.transpose(0, 3, 1, 2))

        embeddings = self._reconhecedor.run(None, {self._entrada_reconhecedor: lote})[0]
        normas = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(normas, 1e-12)

    @staticmethod
    def _rosto_principal(rostos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        No cadastro, o aluno é quem está mais perto da câmara: o maior rosto.
        A confiança do SCRFD não serve para esta escolha — rostos pequenos ao
        fundo podem receber score maior do que o rosto principal.
        """
        return max(rostos, key=lambda r: r["facial_area"]["w"] * r["facial_area"]["h"])

    # ------------------------------------------------------------------
    # Interface usada pelo AlunoService
    # ------------------------------------------------------------------

    def process_single_face(self, img: np.ndarray) -> List[float]:
        """
        Processa uma foto única para registo. Extrai o embedding do rosto principal.
        Garante que deteções com confiança <= LIMIAR_DETECCAO são descartadas.
        """
        self._carregar_modelos()

        rostos = self._detectar(img)
        if not rostos:
            raise ValueError("Nenhum rosto detetado na imagem.")

        best_face = self._rosto_principal(rostos)

        if best_face["confidence"] <= self.confidence_threshold:
            raise ValueError(f"Confiança do rosto detetado ({best_face['confidence']}) é inferior ou igual ao limite exigido ({self.confidence_threshold}). Falso positivo descartado.")

        return self._vetorizar(img, [best_face])[0].tolist()

    def process_multiple_faces(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Processa uma foto densa de uma sala de aula.
        Descarta deteções de rosto com confiança <= LIMIAR_DETECCAO.
        Retorna uma lista contendo os embeddings e a bounding box.
        """
        self._carregar_modelos()

        rostos = [r for r in self._detectar(img) if r["confidence"] > self.confidence_threshold]
        if not rostos:
            return []

        embeddings = self._vetorizar(img, rostos)

        return [
            {
                "embedding": embedding.tolist(),
                "facial_area": rosto["facial_area"],
                "relative_area": rosto["relative_area"],
                "confidence": rosto["confidence"],
            }
            for rosto, embedding in zip(rostos, embeddings)
        ]
