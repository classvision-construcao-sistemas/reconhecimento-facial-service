"""
Avalia e calibra o motor biométrico com fotos reais, sem tocar no Zilliz.

Tudo é calculado em memória: nenhuma foto ou vetor é gravado na base. Só são
impressos números — o script nunca mostra imagens.

Estrutura esperada da pasta de fotos:

    fotos/
    ├── individuais/<matricula>/*.jpg   2+ fotos por pessoa, um rosto por foto
    ├── turmas/*.jpg                    fotos de grupo (opcional)
    └── gabarito.csv                    foto_turma,matriculas_presentes (a;b;c)

Uso:
    python scripts/avaliar_reconhecimento.py /caminho/fotos --modelo topofr
    python scripts/avaliar_reconhecimento.py /caminho/fotos --modelo deepface   # pipeline antigo

O modo deepface reproduz o pipeline anterior (MTCNN + ArcFace do DeepFace) e
exige o pacote deepface instalado no ambiente — só existe para medir o "antes".
"""
import argparse
import csv
import itertools
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402
from app.util.image_processor import ImageProcessor  # noqa: E402

EXTENSOES = (".jpg", ".jpeg", ".png", ".webp")


# ---------------------------------------------------------------------------
# Motores: devolvem [(confiança, embedding normalizado, área)] por foto
# ---------------------------------------------------------------------------

class MotorTopoFR:
    nome = "SCRFD + TopoFR R50"

    def __init__(self):
        from app.service.face_extraction_service import FaceExtractionService
        self.servico = FaceExtractionService()
        self.servico._carregar_modelos()

    def rostos(self, img):
        detectados = self.servico._detectar(img)
        if not detectados:
            return []
        vetores = self.servico._vetorizar(img, detectados)
        return [(r["confidence"], v, r["facial_area"]["w"] * r["facial_area"]["h"])
                for r, v in zip(detectados, vetores)]


class MotorDeepFace:
    nome = "MTCNN + ArcFace (DeepFace, pipeline antigo)"

    def __init__(self):
        from deepface import DeepFace
        self.DeepFace = DeepFace

    def rostos(self, img):
        # o pipeline antigo reduzia a foto a 1024 px antes de tudo
        img = ImageProcessor.redimensionar(img, 1024)
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, img)
            try:
                resultados = self.DeepFace.represent(
                    img_path=tmp.name, model_name="ArcFace",
                    detector_backend="mtcnn", enforce_detection=False,
                )
            except Exception:
                return []
        saida = []
        for r in resultados:
            if r.get("face_confidence", 0) <= 0:
                continue  # enforce_detection=False devolve a imagem inteira quando não acha rosto
            v = np.asarray(r["embedding"], np.float32)
            area = r["facial_area"]["w"] * r["facial_area"]["h"]
            saida.append((float(r["face_confidence"]), v / max(np.linalg.norm(v), 1e-12), area))
        return saida


# ---------------------------------------------------------------------------
# Leitura das fotos
# ---------------------------------------------------------------------------

def ler(caminho):
    img = cv2.imread(caminho)
    if img is None:
        raise ValueError(f"não consegui abrir {caminho}")
    return ImageProcessor.redimensionar(img, settings.LARGURA_MAXIMA_ORIGINAL)


def fotos_individuais(base):
    """{identidade: [caminhos]}. Ficheiros soltos contam como identidade própria."""
    pasta = os.path.join(base, "individuais")
    pessoas = {}
    for nome in sorted(os.listdir(pasta)) if os.path.isdir(pasta) else []:
        caminho = os.path.join(pasta, nome)
        if os.path.isdir(caminho):
            fotos = [os.path.join(caminho, f) for f in sorted(os.listdir(caminho)) if f.lower().endswith(EXTENSOES)]
            if fotos:
                pessoas[nome] = fotos
        elif nome.lower().endswith(EXTENSOES):
            pessoas[os.path.splitext(nome)[0]] = [caminho]
    return pessoas


def ler_gabarito(base):
    caminho = os.path.join(base, "gabarito.csv")
    if not os.path.exists(caminho):
        return {}
    with open(caminho, newline="") as f:
        return {
            linha["foto_turma"].strip(): {m.strip() for m in linha["matriculas_presentes"].split(";") if m.strip()}
            for linha in csv.DictReader(f)
        }


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def resumo(valores):
    v = np.asarray(valores)
    return f"n={len(v):<4} min={v.min():.3f}  média={v.mean():.3f}  máx={v.max():.3f}"


def taxas(genuinos, impostores, limiar):
    frr = np.mean(np.asarray(genuinos) < limiar) if genuinos else float("nan")
    far = np.mean(np.asarray(impostores) >= limiar) if impostores else float("nan")
    return far, frr


def sugerir_limiar(genuinos, impostores):
    """
    Limiar que minimiza FAR + FRR. Quando vários empatam (por exemplo, quando
    as duas distribuições não se sobrepõem), devolve o meio desse intervalo:
    a ponta ficaria colada a um dos lados e erraria à primeira foto nova.
    """
    candidatos = np.linspace(0.0, 1.0, 1001)
    custos = np.array([sum(taxas(genuinos, impostores, c)) for c in candidatos])
    melhores = candidatos[custos == custos.min()]
    return float((melhores.min() + melhores.max()) / 2)


def aumentos(img):
    """Variações da mesma foto. Dão um teto otimista de robustez, não calibração."""
    h, w = img.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0)
    pequena = cv2.resize(img, (max(1, int(w * 0.35)), max(1, int(h * 0.35))), interpolation=cv2.INTER_AREA)
    return {
        "espelhada": cv2.flip(img, 1),
        "mais escura (-40%)": cv2.convertScaleAbs(img, alpha=0.6, beta=0),
        "rodada 10°": cv2.warpAffine(img, rot, (w, h)),
        "baixa resolução (35%)": cv2.resize(pequena, (w, h), interpolation=cv2.INTER_LINEAR),
    }


# ---------------------------------------------------------------------------

def principal(rostos):
    """Mesma regra do cadastro no serviço: o maior rosto da foto."""
    return max(rostos, key=lambda r: r[2]) if rostos else None


def main():
    parser = argparse.ArgumentParser(description="Avalia e calibra o reconhecimento facial com fotos reais.")
    parser.add_argument("fotos", help="pasta com individuais/, turmas/ e gabarito.csv")
    parser.add_argument("--modelo", choices=["topofr", "deepface"], default="topofr")
    parser.add_argument("--limiar", type=float, default=settings.LIMIAR_RECONHECIMENTO,
                        help="limiar de reconhecimento a avaliar nas turmas (padrão: o configurado)")
    parser.add_argument("--limiar-deteccao", type=float, default=settings.LIMIAR_DETECCAO)
    args = parser.parse_args()

    motor = MotorTopoFR() if args.modelo == "topofr" else MotorDeepFace()
    print(f"\nMotor: {motor.nome}")
    print(f"Limiares: deteção > {args.limiar_deteccao} | reconhecimento >= {args.limiar}\n")

    # --- fotos individuais: um embedding (rosto principal) por foto ---
    pessoas = fotos_individuais(args.fotos)
    vetores = {}
    for pessoa, fotos in pessoas.items():
        for foto in fotos:
            rosto = principal(motor.rostos(ler(foto)))
            if rosto is None or rosto[0] <= args.limiar_deteccao:
                print(f"  [aviso] sem rosto válido em {os.path.relpath(foto, args.fotos)}")
                continue
            vetores.setdefault(pessoa, []).append(rosto[1])

    print(f"Individuais: {sum(len(v) for v in vetores.values())} foto(s) de {len(vetores)} pessoa(s)")

    genuinos = [float(a @ b) for vs in vetores.values() for a, b in itertools.combinations(vs, 2)]
    impostores = [float(a @ b) for (p1, v1), (p2, v2) in itertools.combinations(vetores.items(), 2) for a in v1 for b in v2]

    if genuinos:
        print(f"  mesma pessoa       : {resumo(genuinos)}")
    if impostores:
        print(f"  pessoas diferentes : {resumo(impostores)}")
    if genuinos and impostores:
        for limiar in sorted({args.limiar, 0.5, 0.75}):
            far, frr = taxas(genuinos, impostores, limiar)
            print(f"  limiar {limiar:.2f}: aceita impostor {far:6.1%} | rejeita a própria pessoa {frr:6.1%}")
        print(f"  >>> limiar sugerido (menor FAR+FRR): {sugerir_limiar(genuinos, impostores):.3f}")
    else:
        print("  (para calibrar são precisas 2+ fotos da mesma pessoa e 2+ pessoas)")

    # --- robustez com variações da mesma foto (funciona mesmo com 1 foto) ---
    print("\nRobustez (mesma foto com variações — teto otimista, NÃO é calibração):")
    for pessoa, fotos in pessoas.items():
        img = ler(fotos[0])
        base = principal(motor.rostos(img))
        if base is None:
            continue
        linha = []
        for nome, variante in aumentos(img).items():
            r = principal(motor.rostos(variante))
            linha.append(f"{nome}: {'sem rosto' if r is None else f'{float(base[1] @ r[1]):.3f}'}")
        print(f"  {pessoa}: " + " | ".join(linha))

    # --- turmas: simula cadastro (1ª foto de cada pessoa) + auditoria ---
    gabarito = ler_gabarito(args.fotos)
    pasta_turmas = os.path.join(args.fotos, "turmas")
    turmas = sorted(f for f in os.listdir(pasta_turmas) if f.lower().endswith(EXTENSOES)) if os.path.isdir(pasta_turmas) else []
    if not turmas:
        print("\nTurmas: nenhuma foto em turmas/.")
        return

    galeria = {p: vs[0] for p, vs in vetores.items()}
    nomes = list(galeria)
    matriz = np.stack([galeria[p] for p in nomes]) if nomes else np.zeros((0, 512))

    print(f"\nTurmas (galeria = 1ª foto de cada uma das {len(nomes)} pessoa(s)):")
    total = {"certos": 0, "errados": 0, "perdidos": 0, "presentes": 0}
    for turma in turmas:
        rostos = motor.rostos(ler(os.path.join(pasta_turmas, turma)))
        por_deteccao = {t: sum(c > t for c, *_ in rostos) for t in (0.4, 0.5, 0.6)}
        validos = [v for c, v, _ in rostos if c > args.limiar_deteccao]

        identificados = {}
        for v in validos:
            if not nomes:
                break
            sims = matriz @ v
            i = int(np.argmax(sims))
            if sims[i] >= args.limiar:
                identificados[nomes[i]] = max(identificados.get(nomes[i], 0), float(sims[i]))

        print(f"  {turma}: rostos por limiar de deteção {por_deteccao} | válidos={len(validos)} | identificados={len(identificados)}")
        presentes = gabarito.get(turma)
        if presentes is None:
            print("     (sem linha no gabarito.csv — não dá para contar acertos)")
            continue
        certos = {p for p in identificados if p in presentes}
        errados = set(identificados) - presentes
        perdidos = (presentes & set(nomes)) - set(identificados)
        print(f"     certos={len(certos)}  identificação errada={len(errados)} {sorted(errados) or ''}  não encontrados={len(perdidos)} {sorted(perdidos) or ''}")
        for k, v in (("certos", len(certos)), ("errados", len(errados)), ("perdidos", len(perdidos)), ("presentes", len(presentes & set(nomes)))):
            total[k] += v

    if total["presentes"]:
        print(f"\n  TOTAL: encontrou {total['certos']}/{total['presentes']} alunos cadastrados presentes "
              f"({total['certos'] / total['presentes']:.0%}) | identificações erradas: {total['errados']}")


if __name__ == "__main__":
    main()
