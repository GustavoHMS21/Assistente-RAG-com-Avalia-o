"""Carrega configurações locais sem colocar segredos dentro do código."""

# Permite usar recursos modernos de tipagem sem avaliá-los imediatamente.
from __future__ import annotations

# Importa ferramentas para variáveis de ambiente e caminhos de arquivos.
import os
from pathlib import Path

# Importa o carregador responsável por ler as linhas do arquivo .env.
from dotenv import load_dotenv


# Descobre a pasta principal e o arquivo privado de configuração do projeto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def load_environment(env_path: Path = DEFAULT_ENV_PATH) -> None:
    """Carrega o .env sem substituir uma variável já definida no Windows."""

    # O parâmetro override=False preserva uma configuração externa existente.
    load_dotenv(dotenv_path=env_path, override=False)


# Endereço padrão do servidor Ollama quando executado localmente.
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def get_ollama_host(env_path: Path = DEFAULT_ENV_PATH) -> str:
    """Obtém o endereço do servidor Ollama, permitindo troca via .env."""

    # Carrega o arquivo privado antes de consultar a variável.
    load_environment(env_path)

    # Usa o endereço local padrão quando nada for configurado.
    return os.getenv("OLLAMA_HOST", "").strip() or DEFAULT_OLLAMA_HOST
