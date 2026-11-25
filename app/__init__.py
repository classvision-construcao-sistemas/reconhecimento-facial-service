import os
import re

ORIGENS_LOCAIS = "http://localhost:[*],http://127.0.0.1:[*]"


def origens_permitidas(valor=None):
    """
    Converte CORS_ALLOWED_ORIGINS (lista separada por vírgula, como nos
    serviços Java) em expressões regulares para o flask-cors. "[*]" é
    qualquer porta e "*" qualquer trecho sem barra, ex.:
    "http://localhost:[*],https://classvision.vercel.app".
    """
    if valor is None:
        valor = os.getenv("CORS_ALLOWED_ORIGINS") or ORIGENS_LOCAIS
    origens = []
    for origem in (item.strip() for item in valor.split(",")):
        if not origem:
            continue
        padrao = re.escape(origem.rstrip("/"))
        padrao = padrao.replace(re.escape(":[*]"), r"(:\d+)?")
        padrao = padrao.replace(re.escape("*"), r"[^/]*")
        # O flask-cors usa re.match: o "$" impede "https://app.com.evil.com".
        origens.append(padrao + "$")
    return origens


def create_app():
    # pyrefly: ignore [missing-import]
    from flask import Flask
    # pyrefly: ignore [missing-import]
    from flask_cors import CORS

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
    app.json.ensure_ascii = False

    # Origens do app no navegador: as locais (Flutter Web em desenvolvimento)
    # e as de CORS_ALLOWED_ORIGINS, no mesmo formato dos serviços Java.
    CORS(app, resources={r"/api/*": {"origins": origens_permitidas()}})

    from app.controller.aluno_controller import aluno_bp
    from app.controller.docs_controller import docs_bp

    app.register_blueprint(
        aluno_bp,
        url_prefix="/api"
    )

    # Swagger UI em /apidocs e especificação em /openapi.yaml
    app.register_blueprint(docs_bp)

    return app
