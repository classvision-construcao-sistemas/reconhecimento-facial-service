def create_app():
    # pyrefly: ignore [missing-import]
    from flask import Flask
    # pyrefly: ignore [missing-import]
    from flask_cors import CORS

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
    app.json.ensure_ascii = False

    # O Flutter Web usa uma porta local diferente da API durante o
    # desenvolvimento. Autoriza somente origens locais no navegador.
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": [
                    r"http://localhost(:\d+)?",
                    r"http://127\.0\.0\.1(:\d+)?",
                ]
            }
        },
    )

    from app.controller.aluno_controller import aluno_bp
    from app.controller.docs_controller import docs_bp

    app.register_blueprint(
        aluno_bp,
        url_prefix="/api"
    )

    # Swagger UI em /apidocs e especificação em /openapi.yaml
    app.register_blueprint(docs_bp)

    return app
