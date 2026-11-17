import functools
import logging

# pyrefly: ignore [missing-import]
import jwt
# pyrefly: ignore [missing-import]
from flask import g, jsonify, request

from app.config import settings


class ConfiguracaoJwtError(RuntimeError):
    """JWT_SECRET ausente ou fraco: falha de configuração do servidor."""


def _chave():
    segredo = settings.JWT_SECRET
    if not segredo:
        raise ConfiguracaoJwtError(
            "JWT_SECRET não definido. Use o mesmo segredo do cv-authentication-service no .env."
        )
    chave = segredo.encode("utf-8")
    if len(chave) < 32:
        raise ConfiguracaoJwtError("JWT_SECRET deve ter pelo menos 32 bytes.")
    return chave


def _algoritmo_para(chave):
    """
    Mesma regra do turma-service (e do Keys.hmacShaKeyFor do jjwt, usado pelo
    cv-authentication-service para assinar): o algoritmo é fixado pelo
    tamanho da chave, e só esse é aceito.
    """
    if len(chave) >= 64:
        return "HS512"
    if len(chave) >= 48:
        return "HS384"
    return "HS256"


def _erro(status, mensagem):
    resposta = jsonify({"error": mensagem})
    resposta.status_code = status
    if status == 401:
        resposta.headers["WWW-Authenticate"] = "Bearer"
    return resposta


def requer_perfis(*perfis):
    """
    Exige um JWT válido no cabeçalho 'Authorization: Bearer <token>' com pelo
    menos um dos perfis indicados na claim 'roles' (formato ROLE_<PERFIL>,
    como o Spring Security). Responde 401 sem token válido e 403 sem perfil.
    As claims do token ficam disponíveis em flask.g.usuario.
    """
    exigidos = {f"ROLE_{perfil}" for perfil in perfis}

    def decorador(rota):
        @functools.wraps(rota)
        def protegida(*args, **kwargs):
            try:
                chave = _chave()
            except ConfiguracaoJwtError as e:
                logging.error(str(e))
                return _erro(500, "Autenticação não configurada no servidor.")

            tipo, _, token = request.headers.get("Authorization", "").partition(" ")
            token = token.strip()
            if tipo.lower() != "bearer" or not token:
                return _erro(401, "Token de acesso ausente. Envie o cabeçalho 'Authorization: Bearer <token>'.")

            try:
                claims = jwt.decode(
                    token,
                    chave,
                    algorithms=[_algoritmo_para(chave)],
                    options={"require": ["exp"]},
                    leeway=settings.JWT_TOLERANCIA_SEGUNDOS,
                )
            except jwt.ExpiredSignatureError:
                return _erro(401, "Token expirado.")
            except jwt.InvalidTokenError:
                return _erro(401, "Token inválido.")

            roles = claims.get("roles")
            if not isinstance(roles, list) or not exigidos.intersection(r for r in roles if isinstance(r, str)):
                return _erro(403, "O seu perfil não tem permissão para esta operação.")

            g.usuario = claims
            return rota(*args, **kwargs)

        return protegida

    return decorador
