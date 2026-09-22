"""Testes automáticos da geração e validação dos embeddings."""

# Importa ferramentas para ler JSON, cálculos, caminhos e testes automáticos.
import json
import math
from pathlib import Path
import unittest

# Importa as funções da etapa de embeddings que serão verificadas.
from src.embedder import create_embeddings_payload, prepare_embedding_text


# Descobre a pasta principal do projeto a partir da localização deste teste.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeEmbeddingModel:
    """Simula um modelo pequeno para testar a lógica sem realizar downloads."""

    def encode(self, texts, **kwargs):
        """Produz vetores normalizados e determinísticos com três dimensões."""

        # Guarda os dados recebidos para permitir a conferência pelo teste.
        self.received_texts = texts
        self.received_options = kwargs

        # Alterna entre vetores unitários, todos com comprimento matemático igual a 1.
        base_vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        return [base_vectors[index % 3] for index in range(len(texts))]


class PrepareEmbeddingTextTests(unittest.TestCase):
    """Verifica a preparação do texto enviado ao modelo."""

    def test_adds_title_category_and_content(self):
        """Confirma que o modelo recebe contexto além do texto isolado."""

        # Cria um chunk pequeno com os campos usados na preparação.
        chunk = {
            "chunk_id": "TESTE-001",
            "document_title": "Política de Férias",
            "category": "Férias",
            "text": "O colaborador pode converter até 10 dias.",
        }

        # Prepara o texto e verifica se as três informações foram incluídas.
        prepared = prepare_embedding_text(chunk)
        self.assertIn("Política de Férias", prepared)
        self.assertIn("Categoria: Férias", prepared)
        self.assertIn("converter até 10 dias", prepared)


class EmbeddingIntegrationTests(unittest.TestCase):
    """Verifica a geração completa usando os chunks reais e um modelo simulado."""

    def test_creates_one_normalized_vector_per_chunk(self):
        """Confirma quantidade, dimensão, origem e normalização dos vetores."""

        # Carrega os 27 chunks verdadeiros sem depender de internet ou modelo pesado.
        input_path = PROJECT_ROOT / "data" / "processed" / "policy_chunks.json"
        with input_path.open("r", encoding="utf-8") as stream:
            chunks_payload = json.load(stream)

        # Gera embeddings previsíveis usando o modelo simulado.
        fake_model = FakeEmbeddingModel()
        payload = create_embeddings_payload(
            chunks_payload,
            model=fake_model,
            model_name="modelo-de-teste",
            batch_size=8,
        )

        # Confere se cada chunk recebeu exatamente um vetor de três dimensões.
        self.assertEqual(payload["summary"]["chunk_count"], 27)
        self.assertEqual(payload["summary"]["embedding_count"], 27)
        self.assertEqual(payload["summary"]["dimension"], 3)

        # Confere a preservação dos metadados e a normalização de cada resultado.
        for record in payload["embeddings"]:
            self.assertTrue(record["chunk_id"])
            self.assertTrue(record["text"])
            self.assertEqual(len(record["embedding"]), 3)
            vector_norm = math.sqrt(sum(value * value for value in record["embedding"]))
            self.assertAlmostEqual(vector_norm, 1.0)

        # Confere se a opção de normalização foi realmente enviada ao modelo.
        self.assertTrue(fake_model.received_options["normalize_embeddings"])


# Permite executar os testes diretamente por este arquivo.
if __name__ == "__main__":
    unittest.main()
