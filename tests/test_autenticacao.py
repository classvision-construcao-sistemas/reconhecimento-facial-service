import io
import time

import jwt
import pytest

from app.config import settings
from tests.conftest import SEGREDO_TESTE


ROTAS_PROTEGIDAS = [
    ("post", "/api/v1/cadastrar-aluno"),
    ("get", "/api/v1/alunos/2024001"),
    ("delete", "/api/v1/alunos/2024001"),
    ("post", "/api/v1/auditar-sala"),
]


def chamar(cliente, metodo, rota, token=None, cabecalho=None, **kwargs):
    headers = {}
    if cabecalho is not None:
        headers["Authorization"] = cabecalho
    elif token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return getattr(cliente, metodo)(rota, headers=headers, **kwargs)


def imagem(imagem_png):
    return {"image": (io.BytesIO(imagem_png), "foto.png")}


# ---------------------------------------------------------------------------
# Sem token / token inválido -> 401
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metodo,rota", ROTAS_PROTEGIDAS)
def test_rota_protegida_sem_token_responde_401(client_anonimo, metodo, rota):
    resposta = chamar(client_anonimo, metodo, rota)

    assert resposta.status_code == 401
    assert resposta.headers["WWW-Authenticate"] == "Bearer"
    assert "Authorization" in resposta.get_json()["error"]


@pytest.mark.parametrize("cabecalho", ["Basic dXNlcjpzZW5oYQ==", "Bearer", "Bearer   ", "token-sem-esquema"])
def test_cabecalho_mal_formado_responde_401(client_anonimo, cabecalho):
    assert chamar(client_anonimo, "get", "/api/v1/alunos/1", cabecalho=cabecalho).status_code == 401


@pytest.mark.parametrize("esquema", ["Basic", "Token", "JWT"])
def test_token_valido_com_esquema_diferente_de_bearer_responde_401(client_anonimo, gerar_token, esquema):
    assert chamar(client_anonimo, "get", "/api/v1/alunos/1", cabecalho=f"{esquema} {gerar_token()}").status_code == 401


def test_token_com_assinatura_de_outra_chave_responde_401(client_anonimo, gerar_token):
    token = gerar_token(segredo="outro-segredo-qualquer-com-mais-de-sessenta-e-quatro-bytes-de-tamanho!!")

    resposta = chamar(client_anonimo, "get", "/api/v1/alunos/1", token)

    assert resposta.status_code == 401
    assert resposta.get_json()["error"] == "Token inválido."


def test_token_adulterado_responde_401(client_anonimo, gerar_token):
    cabecalho, carga, assinatura = gerar_token(perfis=("ASSISTENTE",)).split(".")
    falsa = jwt.encode({"roles": ["ROLE_ADMINISTRADOR"], "exp": int(time.time()) + 60}, "x" * 64, algorithm="HS512")
    token = ".".join([cabecalho, falsa.split(".")[1], assinatura])

    assert chamar(client_anonimo, "get", "/api/v1/alunos/1", token).status_code == 401


def test_token_com_alg_none_responde_401(client_anonimo):
    token = jwt.encode({"roles": ["ROLE_ADMINISTRADOR"], "exp": int(time.time()) + 60}, None, algorithm="none")

    assert chamar(client_anonimo, "get", "/api/v1/alunos/1", token).status_code == 401


def test_token_assinado_com_algoritmo_diferente_do_esperado_responde_401(client_anonimo, gerar_token):
    # mesma chave, mas HS256: com 64+ bytes só HS512 é aceito (regra do turma-service)
    token = gerar_token(algoritmo="HS256")

    assert chamar(client_anonimo, "get", "/api/v1/alunos/1", token).status_code == 401


def test_token_expirado_responde_401(client_anonimo, gerar_token):
    token = gerar_token(expira_em=-120)

    resposta = chamar(client_anonimo, "get", "/api/v1/alunos/1", token)

    assert resposta.status_code == 401
    assert resposta.get_json()["error"] == "Token expirado."


def test_tolerancia_de_relogio_de_60_segundos(client_anonimo, gerar_token):
    # igual ao Spring Security: expirado há 30 s ainda é aceito
    token = gerar_token(expira_em=-30)

    assert chamar(client_anonimo, "get", "/api/v1/alunos/1", token).status_code == 404


def test_token_sem_exp_responde_401(client_anonimo, gerar_token):
    token = gerar_token(exp=...)

    assert chamar(client_anonimo, "get", "/api/v1/alunos/1", token).status_code == 401


# ---------------------------------------------------------------------------
# Perfis -> 403 / autorizado
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("perfil", settings.PERFIS_GESTAO_ALUNOS)
def test_perfis_de_gestao_podem_cadastrar(client_anonimo, gerar_token, imagem_png, perfil):
    dados = {**imagem(imagem_png), "student_id": f"M-{perfil[:10]}"}

    resposta = chamar(client_anonimo, "post", "/api/v1/cadastrar-aluno", gerar_token(perfis=(perfil,)), data=dados)

    assert resposta.status_code == 201


@pytest.mark.parametrize("perfil", ["PROFESSOR", "ASSISTENTE"])
def test_professor_e_assistente_nao_podem_cadastrar_nem_excluir(client_anonimo, gerar_token, imagem_png, perfil):
    token = gerar_token(perfis=(perfil,))
    dados = {**imagem(imagem_png), "student_id": "2024001"}

    cadastro = chamar(client_anonimo, "post", "/api/v1/cadastrar-aluno", token, data=dados)
    exclusao = chamar(client_anonimo, "delete", "/api/v1/alunos/2024001", token)

    assert cadastro.status_code == 403
    assert exclusao.status_code == 403
    assert "permissão" in cadastro.get_json()["error"]


@pytest.mark.parametrize("perfil", settings.PERFIS_CHAMADA)
def test_perfis_de_chamada_podem_auditar(client_anonimo, gerar_token, imagem_png, perfil):
    resposta = chamar(client_anonimo, "post", "/api/v1/auditar-sala", gerar_token(perfis=(perfil,)), data=imagem(imagem_png))

    assert resposta.status_code == 200


@pytest.mark.parametrize("perfil", ["SECRETARIO", "ASSISTENTE"])
def test_secretario_e_assistente_nao_podem_auditar(client_anonimo, gerar_token, imagem_png, perfil):
    resposta = chamar(client_anonimo, "post", "/api/v1/auditar-sala", gerar_token(perfis=(perfil,)), data=imagem(imagem_png))

    assert resposta.status_code == 403


@pytest.mark.parametrize("perfil", settings.PERFIS_CONSULTA)
def test_todos_os_perfis_podem_consultar(client_anonimo, gerar_token, perfil):
    # 404: passou pela autorização e chegou à busca (aluno inexistente)
    assert chamar(client_anonimo, "get", "/api/v1/alunos/1", gerar_token(perfis=(perfil,))).status_code == 404


@pytest.mark.parametrize("roles", [..., "ROLE_ADMINISTRADOR", [], ["ADMINISTRADOR"], ["ROLE_DESCONHECIDO"], [None, 7]])
def test_claim_roles_ausente_ou_invalida_responde_403(client_anonimo, gerar_token, roles):
    token = gerar_token(roles=roles)

    assert chamar(client_anonimo, "get", "/api/v1/alunos/1", token).status_code == 403


def test_token_valido_e_verificado_antes_de_processar_a_imagem(client_anonimo, fake_ia, gerar_token, imagem_png, monkeypatch):
    chamadas = []
    monkeypatch.setattr(fake_ia, "process_single_face", lambda img: chamadas.append(img) or [0.1] * 512)
    def dados():
        return {**imagem(imagem_png), "student_id": "2024001"}

    sem_token = chamar(client_anonimo, "post", "/api/v1/cadastrar-aluno", data=dados())
    sem_perfil = chamar(client_anonimo, "post", "/api/v1/cadastrar-aluno", gerar_token(perfis=("PROFESSOR",)), data=dados())

    assert (sem_token.status_code, sem_perfil.status_code) == (401, 403)
    assert chamadas == []


# ---------------------------------------------------------------------------
# Compatibilidade e configuração
# ---------------------------------------------------------------------------

def test_aceita_token_no_formato_exato_do_cv_authentication_service(client_anonimo, imagem_png):
    # Mesmas claims e cabeçalho que o JwtTokenProvider (jjwt) produz: {"alg": "HS512"}, sem "typ"
    agora = int(time.time())
    claims = {
        "id": 42, "nome": "Lucas Ferreira", "email": "lucas@ifgoiano.edu.br", "role": "COORDENADOR",
        "instituicaoId": 3, "roles": ["ROLE_COORDENADOR"],
        "sub": "lucas@ifgoiano.edu.br", "iat": agora, "exp": agora + 3600,
    }
    token = jwt.encode(claims, SEGREDO_TESTE, algorithm="HS512", headers={"typ": None})
    assert jwt.get_unverified_header(token) == {"alg": "HS512"}

    resposta = chamar(client_anonimo, "post", "/api/v1/auditar-sala", token, data=imagem(imagem_png))

    assert resposta.status_code == 200


def test_esquema_bearer_nao_diferencia_maiusculas(client_anonimo, gerar_token):
    assert chamar(client_anonimo, "get", "/api/v1/alunos/1", cabecalho=f"bearer {gerar_token()}").status_code == 404


@pytest.mark.parametrize("segredo", [None, "", "curto-demais"])
def test_sem_segredo_configurado_nunca_deixa_passar(client_anonimo, gerar_token, monkeypatch, segredo):
    token = gerar_token()
    monkeypatch.setattr(settings, "JWT_SECRET", segredo)

    resposta = chamar(client_anonimo, "get", "/api/v1/alunos/1", token)

    assert resposta.status_code == 500
    assert resposta.get_json()["error"] == "Autenticação não configurada no servidor."


@pytest.mark.parametrize("rota", ["/api/health", "/openapi.yaml", "/apidocs"])
def test_rotas_publicas_nao_exigem_token(client_anonimo, rota):
    assert client_anonimo.get(rota).status_code == 200


def test_preflight_cors_nao_exige_token(client_anonimo):
    resposta = client_anonimo.options(
        "/api/v1/auditar-sala",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization",
        },
    )

    assert resposta.status_code == 200
    assert resposta.headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"
    assert "authorization" in resposta.headers.get("Access-Control-Allow-Headers", "").lower()


def test_origens_cors_vem_da_variavel_de_ambiente():
    import re

    from app import origens_permitidas

    padroes = origens_permitidas("http://localhost:[*], https://classvision.vercel.app,https://*.exemplo.com")

    def permitida(origem):
        return any(re.match(p, origem, re.IGNORECASE) for p in padroes)

    assert permitida("http://localhost:5173")
    assert permitida("http://localhost")
    assert permitida("https://classvision.vercel.app")
    assert permitida("https://app.exemplo.com")
    assert not permitida("https://classvision.vercel.app.evil.com")
    assert not permitida("https://outro.vercel.app")
    assert not permitida("http://localhost.evil.com")
    # Valor vazio explícito: nenhuma origem liberada.
    assert origens_permitidas("") == []
