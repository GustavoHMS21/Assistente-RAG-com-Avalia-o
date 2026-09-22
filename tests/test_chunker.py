"""Testes automáticos da etapa de chunking das políticas."""

# Importa ferramentas para ler JSON, trabalhar com caminhos e criar testes.
import json
from pathlib import Path
import unittest

# Importa as funções que serão verificadas.
from src.chunker import build_chunks, chunk_page_text, validate_chunk_settings


# Descobre a pasta principal do projeto a partir da localização deste teste.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ChunkPageTextTests(unittest.TestCase):
    """Verifica a divisão de uma página isolada."""

    def test_creates_multiple_chunks_with_overlap(self):
        """Confirma o limite de tamanho e a repetição de contexto."""

        # Cria linhas previsíveis para forçar a formação de mais de um chunk.
        text = "\n".join(
            [
                "Primeiro parágrafo explica o direito geral do colaborador.",
                "Segundo parágrafo apresenta as condições para a solicitação.",
                "Terceiro parágrafo informa o prazo necessário para o pedido.",
                "Quarto parágrafo explica como o gestor realiza a aprovação.",
                "Quinto parágrafo registra as exceções que serão avaliadas.",
            ]
        )

        # Executa a divisão com valores pequenos adequados ao texto de teste.
        chunks = chunk_page_text(text, max_chars=200, overlap_chars=70, min_chars=40)

        # Confere se houve divisão e se todos os chunks respeitam o limite.
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(chunk["character_count"] <= 200 for chunk in chunks))

        # Confere se o último segmento do primeiro chunk reaparece no seguinte.
        repeated_line = chunks[0]["text"].splitlines()[-1]
        self.assertIn(repeated_line, chunks[1]["text"])

    def test_rejects_invalid_settings(self):
        """Confirma que uma sobreposição maior que o chunk é recusada."""

        # A configuração abaixo não permitiria avanço seguro entre os chunks.
        with self.assertRaises(ValueError):
            validate_chunk_settings(max_chars=500, overlap_chars=500, min_chars=100)


class ChunkingIntegrationTests(unittest.TestCase):
    """Verifica o chunking completo usando o JSON real das políticas."""

    def test_builds_traceable_chunks_from_all_policies(self):
        """Confirma quantidade, limites, metadados e termos importantes."""

        # Carrega o resultado verdadeiro da etapa anterior do projeto.
        input_path = PROJECT_ROOT / "data" / "processed" / "policy_pages.json"
        with input_path.open("r", encoding="utf-8") as stream:
            extracted_payload = json.load(stream)

        # Cria os chunks com a configuração padrão que será usada pela aplicação.
        payload = build_chunks(extracted_payload)
        chunks = payload["chunks"]

        # Confere se os três documentos e as nove páginas foram processados.
        self.assertEqual(payload["summary"]["document_count"], 3)
        self.assertEqual(payload["summary"]["page_count"], 9)
        self.assertGreater(payload["summary"]["chunk_count"], 9)

        # Confere se cada chunk possui conteúdo, origem e tamanho válido.
        for chunk in chunks:
            self.assertTrue(chunk["text"].strip())
            self.assertLessEqual(chunk["character_count"], 1000)
            self.assertTrue(chunk["document_id"])
            self.assertTrue(chunk["file_name"].endswith(".pdf"))
            self.assertGreaterEqual(chunk["page"], 1)

        # Garante que nenhum identificador foi usado por dois chunks diferentes.
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))

        # Confere se informações importantes continuam presentes após a divisão.
        combined_text = "\n".join(chunk["text"] for chunk in chunks).casefold()
        self.assertIn("converter até 10 dias", combined_text)
        self.assertIn("até dois dias por semana", combined_text)
        self.assertIn("lei nº 13.709/2018", combined_text)


# Permite executar os testes diretamente por este arquivo.
if __name__ == "__main__":
    unittest.main()
