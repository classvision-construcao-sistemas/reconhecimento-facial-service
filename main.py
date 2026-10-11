import logging

from app import create_app


# Configuração global de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Cria a aplicação Flask: o factory registra o blueprint de alunos, que por sua
# vez compõe o ImageProcessor, o FaceExtractionService e o AlunoRepository.
app = create_app()


if __name__ == "__main__":
    logging.info("A inicializar o Microsserviço ClassVision...")

    # Executa a aplicação na porta 5000 (aberta para a rede)
    app.run(host="0.0.0.0", port=5000)
