import os

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv


load_dotenv()


# =========================================================
# CREDENCIAIS ZILLIZ CLOUD
# =========================================================

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
ZILLIZ_COLLECTION = os.getenv("ZILLIZ_COLLECTION", "Aluno")


# =========================================================
# LIMITES DOS CAMPOS (espelham o schema de create_collection.py)
# =========================================================

TAMANHO_MAXIMO_CAMPOS = {
    "matricula": 20,
    "nome": 150,
    "telefone": 20,
    "responsavel": 150,
    "status": 20,
}

# Limite padrão de registos devolvidos nas listagens
LIMITE_LISTAGEM_PADRAO = 100
LIMITE_LISTAGEM_MAXIMO = 1000


def validar_credenciais():
    """
    Falha cedo e com mensagem clara caso o .env não tenha sido criado, em vez de
    deixar o pymilvus estourar com um erro de conexão obscuro. É chamada apenas
    quando o cliente do Milvus é criado (e não no import), para que os testes
    consigam importar a aplicação sem credenciais reais.
    """
    if not ZILLIZ_URI or not ZILLIZ_TOKEN:
        raise RuntimeError(
            "ZILLIZ_URI e/ou ZILLIZ_TOKEN não definidos. "
            "Copie o ficheiro .env.example para .env e preencha as credenciais "
            "do Zilliz Cloud antes de iniciar o serviço."
        )


# =========================================================
# MOTOR BIOMÉTRICO (SCRFD + TopoFR, via onnxruntime)
# =========================================================

# Pasta local onde os modelos .onnx ficam guardados (ignorada pelo git)
MODELS_DIR = os.getenv(
    "MODELS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models"),
)

# Reconhecedor: TopoFR R50 treinado em Glint360K (NeurIPS 2024), exportado para
# ONNX com scripts/exportar_topofr_onnx.py e publicado na Release do repositório.
TOPOFR_ONNX_FICHEIRO = "topofr_r50_glint360k.onnx"
TOPOFR_ONNX_URL = os.getenv(
    "TOPOFR_ONNX_URL",
    "https://github.com/classvision-construcao-sistemas/reconhecimento-facial-service"
    "/releases/download/modelos-topofr-v1/topofr_r50_glint360k.onnx",
)
TOPOFR_ONNX_SHA256 = "7ebc4f3e87f53ec7e00d7c48e5b042e714291ee149ca928fbee11bcf7eb8adc4"

# Detector: SCRFD-10G. O insightface só o publica dentro do pacote buffalo_l,
# por isso o zip é descarregado da fonte oficial e apenas det_10g.onnx é extraído.
SCRFD_ONNX_FICHEIRO = "det_10g.onnx"
SCRFD_ZIP_URL = os.getenv(
    "SCRFD_ZIP_URL",
    "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip",
)
SCRFD_ONNX_SHA256 = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"

# Limiares. ATENÇÃO: os valores padrão foram herdados do pipeline antigo
# (MTCNN + ArcFace do DeepFace) e ainda NÃO foram recalibrados para
# SCRFD + TopoFR — ver scripts/avaliar_reconhecimento.py.
LIMIAR_DETECCAO = float(os.getenv("LIMIAR_DETECCAO", "0.6"))
LIMIAR_RECONHECIMENTO = float(os.getenv("LIMIAR_RECONHECIMENTO", "0.5"))

# A detecção corre numa cópia reduzida da foto (rápido); os rostos são
# recortados da foto em resolução original (preserva detalhe de quem está longe).
LARGURA_DETECCAO = 1024
TAMANHO_ENTRADA_DETECTOR = (640, 640)
# Teto de segurança para a resolução original (limita memória; 4096 px
# preserva fotos de 12 MP de telemóvel sem redução).
LARGURA_MAXIMA_ORIGINAL = 4096


# =========================================================
# AUTORIZAÇÃO (JWT emitido pelo cv-authentication-service)
# =========================================================

# Mesmo segredo HMAC do cv-authentication-service e do turma-service
# (variável JWT_SECRET). Sem ele, as rotas protegidas respondem 500 — o
# serviço nunca deixa passar um pedido por falta de configuração.
JWT_SECRET = os.getenv("JWT_SECRET")

# Tolerância de relógio na validação de exp/iat, igual à do Spring Security.
JWT_TOLERANCIA_SEGUNDOS = 60

# Perfis (RolesEnum do cv-authentication-service) autorizados em cada rota.
# Espelham as regras do turma-service.
PERFIS_GESTAO_ALUNOS = ("ADMINISTRADOR", "ADMIN_INSTITUICAO", "COORDENADOR", "SECRETARIO")
PERFIS_CHAMADA = ("ADMINISTRADOR", "ADMIN_INSTITUICAO", "COORDENADOR", "PROFESSOR")
PERFIS_CONSULTA = ("ADMINISTRADOR", "ADMIN_INSTITUICAO", "COORDENADOR", "PROFESSOR", "SECRETARIO", "ASSISTENTE")
