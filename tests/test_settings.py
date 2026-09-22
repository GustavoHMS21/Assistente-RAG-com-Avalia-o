"""Testes automáticos do carregamento seguro das configurações locais."""

# Importa ferramentas para ambiente, arquivos temporários e testes.
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

# Importa as funções de configuração que serão verificadas.
from src.settings import DEFAULT_OLLAMA_HOST, get_ollama_host


class EnvironmentSettingsTests(unittest.TestCase):
    """Verifica a resolução do endereço do Ollama sem depender de rede."""

    def test_uses_host_from_env_file(self):
        """Confirma a leitura de um endereço customizado em um arquivo temporário."""

        # Cria um diretório isolado para não tocar no .env verdadeiro do usuário.
        with TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text("OLLAMA_HOST=http://localhost:9999\n", encoding="utf-8")

            # Remove temporariamente qualquer variável real herdada pelo processo.
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(get_ollama_host(env_path), "http://localhost:9999")

    def test_falls_back_to_default_host(self):
        """Confirma que a ausência de configuração usa o endereço local padrão."""

        # Cria um .env vazio para representar um projeto ainda não configurado.
        with TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text("", encoding="utf-8")

            # Garante que nenhuma variável externa altere o resultado do teste.
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(get_ollama_host(env_path), DEFAULT_OLLAMA_HOST)


# Permite executar os testes diretamente por este arquivo.
if __name__ == "__main__":
    unittest.main()
