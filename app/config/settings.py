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


# Falha cedo e com mensagem clara caso o .env não tenha sido criado,
# em vez de deixar o pymilvus estourar com um erro de conexão obscuro.
if not ZILLIZ_URI or not ZILLIZ_TOKEN:
    raise RuntimeError(
        "ZILLIZ_URI e/ou ZILLIZ_TOKEN não definidos. "
        "Copie o ficheiro .env.example para .env e preencha as credenciais "
        "do Zilliz Cloud antes de iniciar o serviço."
    )
