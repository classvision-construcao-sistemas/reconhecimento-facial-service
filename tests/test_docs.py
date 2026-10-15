def test_openapi_yaml(client):
    resposta = client.get("/openapi.yaml")
    assert resposta.status_code == 200
    texto = resposta.get_data(as_text=True)
    for rota in ("/api/v1/cadastrar-aluno", "/api/v1/alunos/{matricula}", "/api/v1/auditar-sala"):
        assert rota in texto


def test_swagger_ui(client):
    resposta = client.get("/apidocs")
    assert resposta.status_code == 200
    assert "swagger-ui" in resposta.get_data(as_text=True)
