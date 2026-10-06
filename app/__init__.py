from flask import Flask


def create_app():

    app = Flask(__name__)

    from app.controller.aluno_controller import aluno_bp

    app.register_blueprint(
        aluno_bp,
        url_prefix="/api"
    )

    return app