# Microsserviço de Reconhecimento Facial (ClassVision)

Microsserviço de reconhecimento facial (ClassVision): deteção de múltiplos rostos, extração de embeddings com **DeepFace/ArcFace** e busca por similaridade numa base vetorial **Zilliz Cloud (Milvus)**.

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
├── service/face_extraction_service.py motor de IA (DeepFace + ArcFace)
├── repository/aluno_repository.py     persistência vetorial (pymilvus)
├── model/aluno.py                     entidade Aluno
└── util/image_processor.py            decodificação/redimensionamento (OpenCV)
```

## Configuração

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # preencher ZILLIZ_URI, ZILLIZ_TOKEN, ZILLIZ_COLLECTION
python create_collection.py
```

## Execução

```bash
python main.py
```

Servidor em `http://0.0.0.0:5000`.

Na primeira execução, o DeepFace pode demorar enquanto prepara os modelos.
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

Os testes usam um cliente Milvus em memória e um DeepFace simulado — não precisam
de credenciais do Zilliz, de TensorFlow nem de rede.

```bash
pip install -r requirements-dev.txt
python -m pytest
```
