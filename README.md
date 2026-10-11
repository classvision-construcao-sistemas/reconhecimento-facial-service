# Microsserviço de Reconhecimento Facial (ClassVision)

Microsserviço de reconhecimento facial (ClassVision): deteção de múltiplos rostos, extração de embeddings com **DeepFace/ArcFace** e busca por similaridade numa base vetorial **Zilliz Cloud (Milvus)**.

Funciona como motor biométrico para a automação de chamadas em sala de aula: recebe a foto da turma, identifica os alunos cadastrados e devolve as matrículas ao sistema principal em Java.

## Arquitetura

```
app/
├── __init__.py                        create_app() — registra o blueprint em /api
├── config/settings.py                 credenciais do Zilliz (via .env)
├── controller/aluno_controller.py     camada HTTP (blueprint aluno_bp)
├── service/aluno_service.py           orquestração do domínio
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

## Endpoints

| Método | Rota | Descrição |
| --- | --- | --- |
| POST | `/api/v1/cadastrar-aluno` | Cadastra a biometria de um aluno (`student_id` + `image`, multipart/form-data) |
| POST | `/api/v1/auditar-sala` | Audita uma foto de turma e devolve os alunos reconhecidos (`image`, multipart/form-data) |
