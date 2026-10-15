import io


URL_CADASTRO = "/api/v1/cadastrar-aluno"


def cadastrar(client, form, **campos):
    return client.post(URL_CADASTRO, data=form(**campos), content_type="multipart/form-data")


# ------------------------------- cadastro ----------------------------------

def test_cadastro_com_dados_completos(client, form, fake_milvus):
    resposta = cadastrar(
        client, form,
        student_id="2024001", nome="Maria Silva", telefone="64999990000",
        responsavel="João Silva", status="ativo",
    )

    assert resposta.status_code == 201
    corpo = resposta.get_json()
    assert corpo["student_id"] == "2024001"
    assert corpo["aluno"]["nome"] == "Maria Silva"

    assert len(fake_milvus.registos) == 1
    registo = fake_milvus.registos[0]
    assert registo["telefone"] == "64999990000"
    assert registo["responsavel"] == "João Silva"
    assert registo["status"] == "ativo"
    assert len(registo["embedding"]) == 512


def test_cadastro_so_com_matricula_preenche_campos_vazios(client, form, fake_milvus):
    resposta = cadastrar(client, form, student_id="2024002")

    assert resposta.status_code == 201
    assert fake_milvus.registos[0]["nome"] == ""


def test_cadastro_aceita_campo_matricula(client, form):
    assert cadastrar(client, form, matricula="2024003").status_code == 201


def test_cadastro_sem_matricula_400(client, form):
    resposta = cadastrar(client, form, nome="Sem Matrícula")
    assert resposta.status_code == 400
    assert "student_id" in resposta.get_json()["error"]


def test_cadastro_sem_imagem_400(client):
    resposta = client.post(URL_CADASTRO, data={"student_id": "1"}, content_type="multipart/form-data")
    assert resposta.status_code == 400
    assert "image" in resposta.get_json()["error"]


def test_cadastro_imagem_invalida_400(client):
    resposta = client.post(
        URL_CADASTRO,
        data={"student_id": "1", "image": (io.BytesIO(b"nao-e-imagem"), "x.png")},
        content_type="multipart/form-data",
    )
    assert resposta.status_code == 400


def test_cadastro_matricula_longa_400(client, form, fake_milvus):
    resposta = cadastrar(client, form, student_id="1" * 21)
    assert resposta.status_code == 400
    assert "matricula" in resposta.get_json()["error"]
    assert fake_milvus.registos == []


def test_cadastro_nome_longo_400(client, form, fake_milvus):
    resposta = cadastrar(client, form, student_id="2024004", nome="a" * 151)
    assert resposta.status_code == 400
    assert "nome" in resposta.get_json()["error"]
    assert fake_milvus.registos == []


def test_cadastro_duplicado_409(client, form, fake_milvus):
    assert cadastrar(client, form, student_id="2024005").status_code == 201

    resposta = cadastrar(client, form, student_id="2024005", nome="Outro")

    assert resposta.status_code == 409
    assert "2024005" in resposta.get_json()["error"]
    assert len(fake_milvus.registos) == 1


def test_cadastro_sem_rosto_400(client, form, fake_ia, monkeypatch):
    def sem_rosto(img):
        raise ValueError("Nenhum rosto detetado na imagem.")

    monkeypatch.setattr(fake_ia, "process_single_face", sem_rosto)
    resposta = cadastrar(client, form, student_id="2024006")
    assert resposta.status_code == 400


# --------------------------- busca e exclusão ------------------------------

def test_buscar_aluno(client, form):
    cadastrar(client, form, student_id="2024010", nome="Ana")

    resposta = client.get("/api/v1/alunos/2024010")

    assert resposta.status_code == 200
    corpo = resposta.get_json()
    assert corpo["nome"] == "Ana"
    assert "embedding" not in corpo


def test_buscar_aluno_inexistente_404(client):
    assert client.get("/api/v1/alunos/999").status_code == 404


def test_excluir_aluno(client, form, fake_milvus):
    cadastrar(client, form, student_id="2024011")

    resposta = client.delete("/api/v1/alunos/2024011")

    assert resposta.status_code == 200
    assert fake_milvus.registos == []
    assert client.get("/api/v1/alunos/2024011").status_code == 404


def test_excluir_aluno_inexistente_404(client):
    assert client.delete("/api/v1/alunos/999").status_code == 404


def test_recadastro_apos_exclusao(client, form):
    cadastrar(client, form, student_id="2024012")
    client.delete("/api/v1/alunos/2024012")
    assert cadastrar(client, form, student_id="2024012").status_code == 201


# ------------------------------- auditoria ---------------------------------

def test_auditar_sala_reconhece_aluno(client, form):
    cadastrar(client, form, student_id="2024020")

    resposta = client.post(URL_CADASTRO.replace("cadastrar-aluno", "auditar-sala"), data=form(), content_type="multipart/form-data")

    assert resposta.status_code == 200
    corpo = resposta.get_json()
    assert corpo["total_faces_detected"] == 1
    assert corpo["recognized_students"][0]["student_id"] == "2024020"


def test_auditar_sala_sem_imagem_400(client):
    assert client.post("/api/v1/auditar-sala", data={}, content_type="multipart/form-data").status_code == 400
