import io
import zipfile

import numpy as np
import pytest

from app.service.face_extraction_service import FaceExtractionService
from app.util import model_downloader


class FakeDetector:
    """Imita SCRFD.detect: bboxes (N,5) [x1,y1,x2,y2,score] e kpss (N,5,2)."""

    def __init__(self, rostos):
        self.rostos = rostos
        self.recebido = None

    def detect(self, img, max_num=0):
        self.recebido = img.shape
        if not self.rostos:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        bboxes = np.array([r[:5] for r in self.rostos], np.float32)
        kpss = np.array([[[r[0] + 5, r[1] + 5]] * 5 for r in self.rostos], np.float32)
        return bboxes, kpss


class FakeSessao:
    """Imita onnxruntime: devolve vetores NÃO normalizados, um por recorte."""

    def __init__(self):
        self.lotes = []

    def run(self, _saidas, entradas):
        lote = entradas["data"]
        self.lotes.append(lote)
        # cada vetor codifica o valor do recorte, para identificar qual rosto foi vetorizado
        return [np.stack([np.full(512, 3.0 + float(x[0, 0, 0])) for x in lote])]


@pytest.fixture
def servico(monkeypatch):
    s = FaceExtractionService()
    s.confidence_threshold = 0.6
    s._reconhecedor = FakeSessao()
    s._entrada_reconhecedor = "data"
    s.alinhados = []

    def alinhar(img, landmarks):
        s.alinhados.append(np.asarray(landmarks).copy())
        valor = int(landmarks[0][0]) % 200
        return np.full((112, 112, 3), valor, np.uint8)

    monkeypatch.setattr(s, "_alinhar", alinhar)
    return s


def foto(largura=800, altura=600):
    return np.zeros((altura, largura, 3), np.uint8)


def test_auditoria_filtra_por_confianca_e_mantem_contrato(servico):
    servico._detector = FakeDetector([
        (10, 20, 60, 80, 0.90),
        (100, 20, 140, 70, 0.61),
        (200, 20, 230, 60, 0.50),
    ])

    faces = servico.process_multiple_faces(foto())

    assert len(faces) == 2
    for face in faces:
        assert set(face) == {"embedding", "facial_area", "confidence"}
        assert set(face["facial_area"]) == {"x", "y", "w", "h"}
        assert all(isinstance(v, int) for v in face["facial_area"].values())
        assert len(face["embedding"]) == 512
        assert np.linalg.norm(face["embedding"]) == pytest.approx(1.0)
    assert faces[0]["facial_area"] == {"x": 10, "y": 20, "w": 50, "h": 60}


def test_todos_os_rostos_vetorizados_num_unico_lote(servico):
    servico._detector = FakeDetector([(10, 10, 50, 50, 0.9)] * 4)

    servico.process_multiple_faces(foto())

    assert len(servico._reconhecedor.lotes) == 1
    assert servico._reconhecedor.lotes[0].shape == (4, 3, 112, 112)
    assert servico._reconhecedor.lotes[0].dtype == np.float32


def test_detecta_em_copia_reduzida_e_recorta_da_resolucao_original(servico):
    servico._detector = FakeDetector([(100, 100, 150, 160, 0.9)])

    faces = servico.process_multiple_faces(foto(largura=2048, altura=1536))

    # o detector recebe a cópia com 1024 px de largura...
    assert servico._detector.recebido[:2] == (768, 1024)
    # ...o facial_area continua nesse referencial (o mesmo de sempre)...
    assert faces[0]["facial_area"] == {"x": 100, "y": 100, "w": 50, "h": 60}
    # ...mas os landmarks usados no recorte estão na escala original (2x)
    np.testing.assert_allclose(servico.alinhados[0][0], [210.0, 210.0])


def test_bbox_que_sai_da_imagem_e_limitada_as_bordas(servico):
    servico._detector = FakeDetector([(-8, -5, 40, 50, 0.9)])

    faces = servico.process_multiple_faces(foto())

    assert faces[0]["facial_area"] == {"x": 0, "y": 0, "w": 40, "h": 50}


def test_auditoria_sem_rostos_devolve_lista_vazia(servico):
    servico._detector = FakeDetector([])

    assert servico.process_multiple_faces(foto()) == []
    assert servico._reconhecedor.lotes == []


def test_cadastro_sem_rosto_levanta_value_error(servico):
    servico._detector = FakeDetector([])

    with pytest.raises(ValueError, match="Nenhum rosto"):
        servico.process_single_face(foto())


def test_cadastro_com_confianca_baixa_levanta_value_error(servico):
    servico._detector = FakeDetector([(10, 10, 50, 50, 0.55)])

    with pytest.raises(ValueError, match="inferior ou igual"):
        servico.process_single_face(foto())


def test_cadastro_usa_o_rosto_de_maior_confianca(servico):
    servico._detector = FakeDetector([
        (10, 10, 50, 50, 0.70),
        (120, 10, 160, 50, 0.95),
    ])

    embedding = servico.process_single_face(foto())

    assert len(embedding) == 512
    assert len(servico.alinhados) == 1
    # landmark do rosto escolhido: x1 + 5 do rosto com score 0.95
    assert servico.alinhados[0][0][0] == pytest.approx(125.0)


def test_importar_a_aplicacao_nao_carrega_os_modelos(client):
    from app.controller import aluno_controller

    assert aluno_controller.face_extraction_service._detector is None


# ---------------------------------------------------------------------------
# model_downloader
# ---------------------------------------------------------------------------

def _sha(conteudo):
    import hashlib
    return hashlib.sha256(conteudo).hexdigest()


def test_downloader_guarda_o_modelo_quando_o_hash_confere(tmp_path, monkeypatch):
    conteudo = b"modelo-onnx-falso"
    monkeypatch.setattr(model_downloader, "_baixar", lambda url, destino: open(destino, "wb").write(conteudo))

    caminho = model_downloader.garantir_modelo(str(tmp_path), "m.onnx", "http://x/m.onnx", _sha(conteudo))

    assert open(caminho, "rb").read() == conteudo
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.onnx"]


def test_downloader_descarta_ficheiro_com_hash_errado(tmp_path, monkeypatch):
    monkeypatch.setattr(model_downloader, "_baixar", lambda url, destino: open(destino, "wb").write(b"adulterado"))

    with pytest.raises(RuntimeError, match="SHA-256"):
        model_downloader.garantir_modelo(str(tmp_path), "m.onnx", "http://x/m.onnx", _sha(b"original"))

    assert list(tmp_path.iterdir()) == []


def test_downloader_extrai_apenas_o_membro_do_zip(tmp_path, monkeypatch):
    conteudo = b"detector-falso"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("outro.onnx", b"nao-interessa")
        zf.writestr("det_10g.onnx", conteudo)
    monkeypatch.setattr(model_downloader, "_baixar", lambda url, destino: open(destino, "wb").write(buffer.getvalue()))

    caminho = model_downloader.garantir_modelo(
        str(tmp_path), "det_10g.onnx", "http://x/pack.zip", _sha(conteudo), membro_zip="det_10g.onnx"
    )

    assert open(caminho, "rb").read() == conteudo
    assert sorted(p.name for p in tmp_path.iterdir()) == ["det_10g.onnx"]


def test_downloader_nao_descarrega_se_o_modelo_ja_existe(tmp_path, monkeypatch):
    (tmp_path / "m.onnx").write_bytes(b"ja-aqui")

    def nao_devia(*_):
        raise AssertionError("não devia descarregar")

    monkeypatch.setattr(model_downloader, "_baixar", nao_devia)
    assert model_downloader.garantir_modelo(str(tmp_path), "m.onnx", "http://x", "qualquer").endswith("m.onnx")
