import io
import re
import sys
import types

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub do DeepFace: evita instalar TensorFlow e descarregar pesos nos testes.
# Tem de ser registado antes de importar a aplicação.
# ---------------------------------------------------------------------------
if "deepface" not in sys.modules:
    deepface_stub = types.ModuleType("deepface")

    class _DeepFaceStub:
        @staticmethod
        def represent(*args, **kwargs):
            raise RuntimeError("DeepFace não deve ser chamado nos testes.")

    deepface_stub.DeepFace = _DeepFaceStub
    sys.modules["deepface"] = deepface_stub


from app import create_app  # noqa: E402
from app.controller import aluno_controller  # noqa: E402


FILTRO_MATRICULA = re.compile(r'^matricula == "((?:[^"\\]|\\.)*)"$')


def _desescapar(valor):
    return re.sub(r"\\(.)", r"\1", valor)


class FakeMilvusClient:
    """Cliente Milvus em memória com o subconjunto usado pelo repositório."""

    def __init__(self):
        self.registos = []
        self.chamadas = []
        self._proximo_id = 1

    def _filtrar(self, filtro):
        self.chamadas.append(filtro)
        m = FILTRO_MATRICULA.match(filtro)
        if m:
            alvo = _desescapar(m.group(1))
            return [r for r in self.registos if r["matricula"] == alvo]
        if filtro == "matricula != ''":
            return list(self.registos)
        raise AssertionError(f"Filtro inesperado nos testes: {filtro}")

    def insert(self, collection_name, data):
        for registo in data:
            self.registos.append({"id": self._proximo_id, **registo})
            self._proximo_id += 1
        return {"insert_count": len(data)}

    def query(self, collection_name, filter, output_fields, limit=None):
        resultado = [{c: r[c] for c in output_fields} for r in self._filtrar(filter)]
        return resultado[:limit] if limit else resultado

    def delete(self, collection_name, filter):
        alvos = self._filtrar(filter)
        self.registos = [r for r in self.registos if r not in alvos]
        return {"delete_count": len(alvos)}

    def load_collection(self, collection_name):
        pass

    def search(self, collection_name, data, anns_field, limit, output_fields):
        if not self.registos:
            return [[] for _ in data]
        primeiro = self.registos[0]
        return [[{"distance": 0.9, "entity": {"matricula": primeiro["matricula"]}}] for _ in data]


@pytest.fixture
def fake_milvus(monkeypatch):
    cliente = FakeMilvusClient()
    monkeypatch.setattr(aluno_controller.aluno_repository, "_client", cliente)
    return cliente


@pytest.fixture
def fake_ia(monkeypatch):
    """Substitui a extração de embeddings por vetores fixos de 512 floats."""
    servico = aluno_controller.face_extraction_service
    monkeypatch.setattr(servico, "process_single_face", lambda img: [0.1] * 512)
    monkeypatch.setattr(
        servico,
        "process_multiple_faces",
        lambda img: [{"embedding": [0.1] * 512, "facial_area": {"x": 1, "y": 2, "w": 3, "h": 4}, "confidence": 0.99}],
    )
    return servico


@pytest.fixture
def client(fake_milvus, fake_ia):
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def imagem_png():
    ok, buffer = cv2.imencode(".png", np.full((64, 64, 3), 128, np.uint8))
    assert ok
    return buffer.tobytes()


@pytest.fixture
def form(imagem_png):
    def _form(**campos):
        dados = {"image": (io.BytesIO(imagem_png), "aluno.png")}
        dados.update(campos)
        return dados
    return _form
