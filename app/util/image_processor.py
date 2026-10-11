# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from werkzeug.datastructures import FileStorage


class ImageProcessor:
    @staticmethod
    def process_upload(file: FileStorage) -> np.ndarray:
        if not file or file.filename == '':
            raise ValueError("Nenhum ficheiro válido fornecido no form-data.")

        file.seek(0)
        file_bytes = np.frombuffer(file.read(), np.uint8)

        if file_bytes.size == 0:
            raise ValueError("O ficheiro enviado está vazio ou corrompido.")

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Não foi possível descodificar a imagem.")

        # Alivia a carga de memória da CPU antes de enviar para a rede neural
        img = ImageProcessor.redimensionar(img)
        return img

    @staticmethod
    def redimensionar(img: np.ndarray, largura_maxima: int = 1024) -> np.ndarray:
        """Redimensiona imagens gigantes mantendo a proporção."""
        altura, largura = img.shape[:2]
        if largura > largura_maxima:
            proporcao = largura_maxima / float(largura)
            nova_altura = int(altura * proporcao)
            return cv2.resize(img, (largura_maxima, nova_altura), interpolation=cv2.INTER_AREA)
        return img
