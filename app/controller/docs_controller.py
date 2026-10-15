import os

# pyrefly: ignore [missing-import]
from flask import Blueprint, send_file


# Documentação OpenAPI estática (docs/openapi.yaml) servida com o Swagger UI
docs_bp = Blueprint("docs", __name__)

OPENAPI_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "docs", "openapi.yaml")
)

SWAGGER_UI_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>ClassVision - API de Reconhecimento Facial</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({ url: "/openapi.yaml", dom_id: "#swagger-ui" });
  </script>
</body>
</html>
"""


@docs_bp.route("/openapi.yaml", methods=["GET"])
def openapi_spec():
    return send_file(OPENAPI_PATH, mimetype="application/yaml")


@docs_bp.route("/apidocs", methods=["GET"])
@docs_bp.route("/apidocs/", methods=["GET"])
def swagger_ui():
    return SWAGGER_UI_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}
