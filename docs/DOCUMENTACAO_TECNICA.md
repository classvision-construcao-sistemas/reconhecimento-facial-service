# DOCUMENTAÇÃO TÉCNICA - ClassVision

O **ClassVision** é um microsserviço de auditoria visual focado na assiduidade escolar. Este documento descreve o processo de configuração do ambiente, a estrutura de classes, os endpoints expostos e a arquitetura concetual da inteligência artificial aplicada.

---

## 1. Guia de Configuração e Inicialização (Conda)

### 1.1 Criar e Ativar o Ambiente Virtual
```bash
# Cria um novo ambiente conda com Python 3.10
conda create -n classvision_env python=3.10 -y

# Ativa o ambiente criado
conda activate classvision_env
```

### 1.2 Instalar as Dependências
```bash
pip install -r requirements.txt
```

### 1.3 Pré-requisitos de Base de Dados
A persistência vetorial é feita no **Zilliz Cloud** (Milvus gerido), não numa instância local. As credenciais são lidas de variáveis de ambiente através do ficheiro `.env`:

```bash
cp .env.example .env
# preencher ZILLIZ_URI, ZILLIZ_TOKEN e ZILLIZ_COLLECTION
```

A coleção (`Aluno` por omissão) é criada uma única vez com:
```bash
python create_collection.py
```

### 1.4 Iniciar o Servidor
```bash
python main.py
```
O servidor ficará ativo e à escuta em `http://0.0.0.0:5000/`.

---

## 2. Estrutura Modular e Explicação dos Ficheiros/Classes

O código segue uma arquitetura em camadas (controller → service → repository), garantindo a separação de responsabilidades (Separation of Concerns).

```
reconhecimento-facial-service/
├── main.py
├── create_collection.py
├── requirements.txt
├── .env.example
└── app/
    ├── __init__.py                        -> create_app()
    ├── config/settings.py                 -> credenciais Zilliz
    ├── controller/aluno_controller.py     -> blueprint aluno_bp (HTTP)
    ├── service/aluno_service.py           -> orquestração do domínio
    ├── service/face_extraction_service.py -> motor de IA (DeepFace/ArcFace)
    ├── repository/aluno_repository.py     -> persistência vetorial
    ├── model/aluno.py                     -> entidade Aluno
    └── util/image_processor.py            -> decodificação de imagens
```

### `main.py`
Ficheiro de arranque (Entry Point).
- **Função:** Configura o `logging`, invoca o factory `create_app()` e coloca o servidor web Flask a correr na porta 5000.

### `app/__init__.py` -> `create_app()`
Application factory.
- **Função:** Cria a instância Flask e registra o blueprint `aluno_bp` sob o prefixo `/api`, resultando nas rotas `/api/v1/cadastrar-aluno` e `/api/v1/auditar-sala`.

### `app/config/settings.py`
Camada de configuração.
- **Função:** Carrega `ZILLIZ_URI`, `ZILLIZ_TOKEN` e `ZILLIZ_COLLECTION` a partir do `.env` (via `python-dotenv`) e falha imediatamente, com mensagem explícita, caso as credenciais não estejam definidas.

### `app/controller/aluno_controller.py` -> blueprint `aluno_bp`
A camada de interface/rede.
- **Função:** É a porta de entrada da aplicação. Compõe as dependências `ImageProcessor`, `FaceExtractionService`, `AlunoRepository` e `AlunoService`, expõe os endpoints HTTP e é responsável pela validação dos inputs, pelo tratamento de erros HTTP e pela serialização das respostas em JSON.

### `app/service/aluno_service.py` -> Classe `AlunoService`
A camada de orquestração do domínio.
- **Função:** Liga o motor de IA à persistência. Em `cadastrar` extrai o embedding do rosto principal e grava-o associado à matrícula; em `auditar_sala` deteta múltiplos rostos, executa a busca vetorial em lote e monta o payload de auditoria (reconhecidos, contagem de desconhecidos e total de rostos).

### `app/service/face_extraction_service.py` -> Classe `FaceExtractionService`
A camada de inteligência artificial.
- **Função:** É o motor de IA do microsserviço, que encapsula o acesso à biblioteca `DeepFace` (deteção com `detector_backend="mtcnn"`, vetorização com `model_name="ArcFace"`). Implementa `process_single_face` e `process_multiple_faces`, operando em duas vias: extração singular ou auditoria densa. Descarta qualquer deteção com confiança inferior ou igual a `0.6`, evitando falsos positivos antes da vetorização ArcFace.

### `app/repository/aluno_repository.py` -> Classe `AlunoRepository`
A camada de infraestrutura e persistência de dados.
- **Função:** Gere a conexão `MilvusClient` com o Zilliz Cloud. `inserir` armazena o embedding de 512 dimensões e `buscar_faces` faz a busca em lote dos rostos de uma sala, aplicando o limiar de similaridade e traduzindo o resultado bruto do Milvus para o formato consumido pelo serviço. Mantém ainda as operações de CRUD auxiliares (`buscar_por_embedding`, `buscar_por_matricula`, `listar`, `buscar_por_nome`, `atualizar`, `excluir`).

### `app/model/aluno.py` -> Classe `Aluno`
A entidade de domínio.
- **Função:** Representa a linha persistida na coleção. Como o motor biométrico apenas conhece a matrícula e o embedding, os campos biográficos obrigatórios do schema (`nome`, `telefone`, `responsavel`, `status`) assumem string vazia por omissão.

### `app/util/image_processor.py` -> Classe `ImageProcessor`
A camada de utilidade para tratamento binário.
- **Função:** O método estático `process_upload` recebe os bytes de uma imagem via *multipart/form-data* (`FileStorage` do Flask) e converte de forma segura para `numpy.ndarray`, validando a integridade dos binários e redimensionando imagens acima de 1024px de largura para aliviar a carga de memória.

### `create_collection.py`
Script utilitário de bootstrap.
- **Função:** Cria a coleção no Zilliz Cloud (se não existir) com o schema de 7 campos, `id` auto-incremental e índices `AUTOINDEX`, sendo o do campo `embedding` configurado com a métrica `COSINE`.

---

## 3. Documentação de Endpoints

### 3.1 Cadastrar Aluno
- **URL:** `POST /api/v1/cadastrar-aluno`
- **Descrição:** Regista um novo aluno no banco de dados através da extração facial de uma imagem singular do seu rosto.
- **Content-Type:** `multipart/form-data`
- **Payload Esperado:**
  - `student_id` (texto): Identificador único do aluno.
  - `image` (ficheiro): Fotografia nítida do rosto do aluno (PNG, JPG).
- **Códigos HTTP Possíveis:**
  - `201 Created`: Cadastro realizado com sucesso.
  - `400 Bad Request`: Parâmetros em falta, ficheiro corrompido ou rosto não validado pela IA (confiança <= 0.6).
  - `500 Internal Server Error`: Falha inesperada (ex: Zilliz em baixo).
- **Esquema de Resposta (Sucesso 201):**
  ```json
  {
    "message": "Aluno 12345 cadastrado com sucesso.",
    "student_id": "12345"
  }
  ```
- **Exemplo de Chamada cURL:**
  ```bash
  curl -X POST http://localhost:5000/api/v1/cadastrar-aluno \
    -F "student_id=2023001" \
    -F "image=@/caminho/para/foto_perfil.jpg"
  ```

### 3.2 Auditar Sala
- **URL:** `POST /api/v1/auditar-sala`
- **Descrição:** Analisa uma fotografia de uma sala de aula de fundo denso, deteta todos os rostos viáveis (confiança > 0.6) e identifica-os comparando com o registo.
- **Content-Type:** `multipart/form-data`
- **Payload Esperado:**
  - `image` (ficheiro): Fotografia da sala de aula contendo os alunos.
- **Códigos HTTP Possíveis:**
  - `200 OK`: Auditoria realizada (retorna a lista de reconhecidos e a contagem de não reconhecidos).
  - `400 Bad Request`: Imagem em falta ou corrompida.
  - `500 Internal Server Error`: Falha inesperada.
- **Esquema de Resposta (Sucesso 200):**
  ```json
  {
    "message": "Auditoria concluída com sucesso.",
    "recognized_students": [
      {
        "student_id": "2023001",
        "confidence_score": 0.85231,
        "facial_area": {"x": 120, "y": 80, "w": 50, "h": 50}
      }
    ],
    "unrecognized_count": 2,
    "total_faces_detected": 3
  }
  ```
- **Exemplo de Chamada cURL:**
  ```bash
  curl -X POST http://localhost:5000/api/v1/auditar-sala \
    -F "image=@/caminho/para/sala_turma_a.jpg"
  ```

---

## 4. Fluxo Concetual de Informação

Quando um pedido chega ao microsserviço (seja para cadastro ou para auditoria), os bytes em bruto (`multipart/form-data`) são interceptados pelo **aluno_controller** e repassados ao **ImageProcessor**, que utiliza o OpenCV para decodificar os binários numa matriz NumPy tridimensional segura. Essa matriz é encaminhada ao **AlunoService**, que aciona o motor **FaceExtractionService**. Neste ponto entra o detetor do DeepFace, capaz de identificar e alinhar caixas delimitadoras (*bounding boxes*) de rostos mesmo em cenas amplas e densas (como o fundo de uma sala de aula). O serviço descarta instantaneamente qualquer deteção cuja confiança seja igual ou inferior a 0.6. Para os rostos filtrados e alinhados, a imagem recortada (*crop*) alimenta o modelo **ArcFace**, que processa as feições geométricas convertendo-as numa representação numérica altamente compacta — um *embedding* (vetor) de 512 dimensões.

No caso de cadastro, esse vetor é remetido ao **AlunoRepository**, onde é acoplado à `matricula` e guardado na coleção vetorial do Zilliz Cloud — com os campos biográficos preenchidos como string vazia, pois o microsserviço é um motor puramente biométrico. Na auditoria, os múltiplos vetores da sala são enviados em lote ao Milvus; este utiliza o índice `AUTOINDEX` e compara rapidamente os 512 pontos flutuantes através da métrica de **Distância Cosseno** (*Cosine Distance*). Os resultados que atinjam o limiar de similaridade (`0.5`) atestam a identidade e traduzem-se, no fim da cadeia, na confirmação de presença dos alunos na sala; os rostos detetados sem correspondência acima do limiar são contabilizados como desconhecidos e devolvidos ao sistema principal em Java apenas como contagem.
