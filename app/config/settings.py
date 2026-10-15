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
