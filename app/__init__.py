# pyrefly: ignore [missing-import]
from flask import Flask


def create_app():

    app = Flask(__name__)

    from app.controller.aluno_controller import aluno_bp
    from app.controller.docs_controller import docs_bp

    app.register_blueprint(
        aluno_bp,
        url_prefix="/api"
    )

    # Swagger UI em /apidocs e especificação em /openapi.yaml
    app.register_blueprint(docs_bp)

    return app
