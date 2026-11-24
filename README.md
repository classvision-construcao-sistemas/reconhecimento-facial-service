# Microsserviço de Reconhecimento Facial (ClassVision)

Microsserviço de reconhecimento facial (ClassVision): deteção de múltiplos rostos, extração de embeddings com **SCRFD + TopoFR** (via onnxruntime) e busca por similaridade numa base vetorial **Zilliz Cloud (Milvus)**.

Funciona como motor biométrico para a automação de chamadas em sala de aula: recebe a foto da turma, identifica os alunos cadastrados e devolve as matrículas ao sistema principal em Java.

## Arquitetura

```
app/
├── __init__.py                        create_app() — registra o blueprint em /api
├── config/settings.py                 credenciais do Zilliz (via .env)
├── controller/aluno_controller.py     camada HTTP (blueprint aluno_bp)
├── controller/docs_controller.py      Swagger UI (/apidocs) e /openapi.yaml
├── service/aluno_service.py           orquestração do domínio
├── service/exceptions.py              erros de domínio (409/404)
├── service/face_extraction_service.py motor de IA (SCRFD + TopoFR)
├── repository/aluno_repository.py     persistência vetorial (pymilvus)
├── model/aluno.py                     entidade Aluno
└── util/image_processor.py            decodificação/redimensionamento (OpenCV)
```

## Configuração

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # preencher ZILLIZ_URI, ZILLIZ_TOKEN, ZILLIZ_COLLECTION
python create_collection.py   # opcional: o serviço cria a coleção sozinho na 1ª utilização
```

Sem conta no Zilliz Cloud, aponte `ZILLIZ_URI` para um Milvus próprio, sem token:
`http://localhost:19530` (o `docker-compose` da pasta `infra/` do ClassVision sobe um)
ou `./milvus.db` (Milvus Lite, `pip install milvus-lite`, só Linux/macOS).

## Execução

```bash
python main.py
```

Servidor em `http://0.0.0.0:5000`.

Na primeira requisição de cadastro ou auditoria, o serviço descarrega os modelos
para `models/` (~16 MB do detetor SCRFD, extraído do pacote oficial `buffalo_l`
do insightface, e ~166 MB do TopoFR) e confere o SHA-256 de cada um. As
execuções seguintes usam a cópia local.
Confirme o serviço em `http://localhost:5000/api/health`.

## Documentação da API (Swagger)

Com o servidor em execução:

- Swagger UI: `http://localhost:5000/apidocs`
- Especificação OpenAPI 3: `http://localhost:5000/openapi.yaml` (ficheiro `docs/openapi.yaml`)

## Endpoints

| Método | Rota | Descrição |
| --- | --- | --- |
| GET | `/api/health` | Verifica se o serviço iniciou e informa o armazenamento ativo |
| POST | `/api/v1/cadastrar-aluno` | Cadastra biometria e dados do aluno (multipart/form-data) |
| GET | `/api/v1/alunos/<matricula>` | Consulta os dados do aluno (sem embedding) |
| DELETE | `/api/v1/alunos/<matricula>` | Exclui o aluno e a biometria |
| POST | `/api/v1/auditar-sala` | Audita uma foto de turma e devolve os alunos reconhecidos (`image`) |

### Contrato do cadastro — `POST /api/v1/cadastrar-aluno`

`multipart/form-data`:

| Campo | Obrigatório | Tamanho máx. | Observação |
| --- | --- | --- | --- |
| `student_id` | sim | 20 | Matrícula (também aceite como `matricula`) |
| `image` | sim | — | Foto do rosto (JPEG/PNG) |
| `nome` | não | 150 | |
| `telefone` | não | 20 | |
| `responsavel` | não | 150 | |
| `status` | não | 20 | ex.: `ativo` |

Respostas:

- `201` — `{"message": "...", "student_id": "2024001", "aluno": {"matricula", "nome", "telefone", "responsavel", "status"}}`
- `400` — campo obrigatório ausente, tamanho excedido, imagem inválida ou nenhum rosto válido
- `409` — já existe aluno com a matrícula (exclua com `DELETE` antes de recadastrar)
- `500` — erro interno (IA ou Zilliz)

Todos os erros têm o formato `{"error": "mensagem"}`.

```bash
curl -X POST http://localhost:5000/api/v1/cadastrar-aluno \
  -F student_id=2024001 -F nome="Maria Silva" -F telefone=64999990000 \
  -F responsavel="João Silva" -F status=ativo -F image=@foto.jpg
```

## Testes

Os testes usam um cliente Milvus em memória, um motor de IA simulado e tokens JWT
gerados localmente — não precisam de credenciais do Zilliz, dos modelos .onnx,
do cv-authentication-service nem de rede.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Motor biométrico

```
foto ─► SCRFD-10G (deteção + 5 landmarks, numa cópia de 1024 px)
     ─► norm_crop (alinhamento 112×112, recortado da foto em resolução original)
     ─► TopoFR R50 / Glint360K (embedding de 512 dimensões, normalizado)
     ─► Zilliz Cloud (similaridade cosseno)
```

- **TopoFR** — *A Closer Look at Topology Alignment on Face Recognition*, NeurIPS 2024
  ([código](https://github.com/DanJun6737/TopoFR)). O checkpoint PyTorch é convertido
  uma única vez com `scripts/exportar_topofr_onnx.py`; o serviço não usa PyTorch.
- **SCRFD** — detetor do [insightface](https://github.com/deepinsight/insightface).
  Os modelos do insightface são licenciados apenas para uso não comercial e de pesquisa.

Os limiares ficam em `app/config/settings.py` e podem ser sobrepostos pelo `.env`
(`LIMIAR_DETECCAO`, `LIMIAR_RECONHECIMENTO`). **Os valores padrão (0.6 e 0.5) foram
herdados do pipeline antigo e ainda precisam de ser recalibrados** com fotos reais:

```bash
python scripts/avaliar_reconhecimento.py /caminho/fotos --modelo topofr
```

O script corre tudo em memória (não grava nada no Zilliz), mede robustez, sugere o
limiar de reconhecimento a partir de pares da mesma pessoa e de pessoas diferentes,
e conta acertos nas fotos de turma contra um `gabarito.csv`. A estrutura esperada
da pasta está descrita no topo do script.

> Trocar de modelo invalida os embeddings já gravados: cada aluno tem de ser
> recadastrado com foto (`DELETE /api/v1/alunos/<matricula>` e novo cadastro).

## Autenticação

Todas as rotas `/api/v1/*` exigem o **JWT emitido pelo cv-authentication-service**,
no cabeçalho `Authorization: Bearer <token>`. `/api/health`, `/apidocs` e
`/openapi.yaml` são públicos.

A validação segue a do `turma-service`: o mesmo segredo `JWT_SECRET` (configure-o
no `.env`), algoritmo HMAC fixado pelo tamanho da chave (HS512 a partir de 64
bytes), `exp` obrigatório com 60 s de tolerância e perfis lidos da claim `roles`.

| Rota | Perfis autorizados |
| --- | --- |
| `POST /api/v1/cadastrar-aluno` | ADMINISTRADOR, ADMIN_INSTITUICAO, COORDENADOR, SECRETARIO |
| `DELETE /api/v1/alunos/<matricula>` | ADMINISTRADOR, ADMIN_INSTITUICAO, COORDENADOR, SECRETARIO |
| `POST /api/v1/auditar-sala` | ADMINISTRADOR, ADMIN_INSTITUICAO, COORDENADOR, PROFESSOR |
| `GET /api/v1/alunos/<matricula>` | todos os perfis |

Respostas: **401** sem token ou com token inválido/expirado, **403** quando o perfil
não tem permissão e **500** se `JWT_SECRET` não estiver configurado (o serviço nunca
deixa passar um pedido por falta de configuração). Os perfis de cada rota ficam em
`app/config/settings.py`.

No Swagger (`/apidocs`), use o botão **Authorize** e cole o token do login.

