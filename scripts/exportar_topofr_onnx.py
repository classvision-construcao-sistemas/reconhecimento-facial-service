"""
Converte um checkpoint PyTorch do TopoFR (NeurIPS 2024) para ONNX.

Roda UMA vez, fora do serviço, num ambiente que tenha PyTorch. O serviço em si
só usa o .onnx gerado, via onnxruntime — nunca importa torch.

O torch2onnx.py do repositório do TopoFR não serve para isso:
  - chama get_model sem num_classes (o IResNet do TopoFR exige o argumento);
  - usa np.float, removido do NumPy;
  - exporta o forward com phase='train', que devolve os logits de treino em vez
    do embedding.

Uso:
    git clone https://github.com/DanJun6737/TopoFR.git /caminho/TopoFR
    python scripts/exportar_topofr_onnx.py \
        --topofr-repo /caminho/TopoFR \
        --checkpoint  /caminho/Glint360K_R50_TopoFR_9727.pt \
        --network     r50 \
        --output      models/topofr_r50_glint360k.onnx
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch


class TopoFRInferencia(torch.nn.Module):
    """Expõe só o caminho de inferência do TopoFR: (N,3,112,112) -> (N,512)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x, phase='infer')


def carregar_topofr(topofr_repo, checkpoint, network):
    sys.path.insert(0, topofr_repo)
    from backbones import get_model

    # weights_only=True: carrega só tensores, sem executar pickle arbitrário
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)

    # A chave 'weight' é a cabeça de classificação usada no treino
    # (num_classes x 512). Só serve para descobrir num_classes e permitir um
    # load strict — ela não participa do caminho de inferência e não vai para
    # o ONNX.
    num_classes = ckpt['weight'].shape[0]
    net = get_model(network, fp16=False, num_classes=num_classes)
    net.load_state_dict(ckpt, strict=True)
    net.eval()
    return net, num_classes


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, 'rb') as f:
        for bloco in iter(lambda: f.read(1 << 20), b''):
            h.update(bloco)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Exporta o TopoFR para ONNX')
    parser.add_argument('--topofr-repo', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--network', default='r50')
    parser.add_argument('--output', required=True)
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()

    net, num_classes = carregar_topofr(args.topofr_repo, args.checkpoint, args.network)
    print(f"checkpoint carregado: {args.network}, cabeça de treino com {num_classes} classes (descartada)")

    modelo = TopoFRInferencia(net).eval()
    exemplo = torch.zeros(1, 3, 112, 112)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.onnx.export(
        modelo,
        exemplo,
        args.output,
        input_names=['data'],
        output_names=['embedding'],
        dynamic_axes={'data': {0: 'batch'}, 'embedding': {0: 'batch'}},
        opset_version=args.opset,
        dynamo=False,
    )

    # Validação: torch e onnxruntime têm de produzir o mesmo embedding
    import onnxruntime as ort
    sessao = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
    lote = torch.from_numpy(np.random.default_rng(0).uniform(-1, 1, (4, 3, 112, 112)).astype(np.float32))
    with torch.no_grad():
        ref = modelo(lote).numpy()
    out = sessao.run(None, {'data': lote.numpy()})[0]

    cos = np.sum(ref * out, axis=1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(out, axis=1))
    print(f"saída ONNX: {out.shape} | diferença máxima vs torch: {np.abs(ref - out).max():.2e} | cosseno mínimo: {cos.min():.6f}")
    if cos.min() < 0.9999:
        raise SystemExit("ERRO: o ONNX não reproduz o modelo PyTorch.")

    tamanho = os.path.getsize(args.output) / (1 << 20)
    print(f"arquivo: {args.output} ({tamanho:.1f} MB)")
    print(f"sha256 : {sha256(args.output)}")


if __name__ == '__main__':
    main()
