import hashlib
import logging
import os
import tempfile
import zipfile

# pyrefly: ignore [missing-import]
import requests


def sha256_ficheiro(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _baixar(url, destino):
    with requests.get(url, stream=True, timeout=60) as resposta:
        resposta.raise_for_status()
        with open(destino, "wb") as f:
            for bloco in resposta.iter_content(chunk_size=1 << 20):
                f.write(bloco)


def garantir_modelo(pasta, ficheiro, url, sha256_esperado, membro_zip=None):
    """
    Devolve o caminho local de um modelo, descarregando-o na primeira execução.

    O descarregamento vai para um ficheiro temporário e só é movido para o
    destino depois de o SHA-256 conferir, para nunca deixar um modelo
    incompleto ou adulterado no lugar do verdadeiro. Se `membro_zip` for
    indicado, `url` aponta para um .zip e apenas esse membro é extraído.
    """
    destino = os.path.join(pasta, ficheiro)
    if os.path.exists(destino):
        return destino

    os.makedirs(pasta, exist_ok=True)
    logging.info(f"Modelo {ficheiro} não encontrado em {pasta}. A descarregar de {url} ...")

    with tempfile.TemporaryDirectory(dir=pasta) as tmp:
        descarregado = os.path.join(tmp, "download")
        _baixar(url, descarregado)

        candidato = descarregado
        if membro_zip:
            with zipfile.ZipFile(descarregado) as zf:
                nomes = [n for n in zf.namelist() if os.path.basename(n) == membro_zip]
                if not nomes:
                    raise RuntimeError(f"{membro_zip} não existe em {url}.")
                candidato = os.path.join(tmp, membro_zip)
                with zf.open(nomes[0]) as origem, open(candidato, "wb") as saida:
                    while bloco := origem.read(1 << 20):
                        saida.write(bloco)

        obtido = sha256_ficheiro(candidato)
        if obtido != sha256_esperado:
            raise RuntimeError(
                f"SHA-256 de {ficheiro} não confere (esperado {sha256_esperado}, obtido {obtido}). "
                "O ficheiro foi descartado."
            )
        os.replace(candidato, destino)

    logging.info(f"Modelo {ficheiro} pronto em {destino}.")
    return destino
