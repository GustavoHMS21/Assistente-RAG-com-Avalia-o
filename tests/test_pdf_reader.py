"""Testes automáticos da leitura e validação das políticas."""

# Importa ferramentas para trabalhar com caminhos e criar testes automáticos.
from pathlib import Path
import unittest

# Importa as funções do projeto que serão verificadas pelos testes.
from src.pdf_reader import extract_all, normalize_page_text, normalized_for_search


# Descobre a pasta principal do projeto a partir da localização deste teste.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TextNormalizationTests(unittest.TestCase):
    """Verifica as pequenas funções responsáveis pela limpeza do texto."""

    def test_normalize_page_text(self):
        """Confirma que espaços extras são removidos sem juntar linhas."""

        # Simula um texto extraído de PDF com espaços, linha vazia e tabulação.
        raw = "  Primeira   linha  \n\n Segunda\tlinha "

        # Compara o resultado real com o texto limpo esperado.
        self.assertEqual(normalize_page_text(raw), "Primeira linha\nSegunda linha")

    def test_normalized_search_ignores_accents_and_case(self):
        """Confirma que a busca ignora diferenças de acento e maiúsculas."""

        # A palavra em maiúsculas e com acento deve virar uma forma simples de busca.
        self.assertEqual(normalized_for_search("FÉRIAS"), "ferias")


class PdfExtractionIntegrationTests(unittest.TestCase):
    """Verifica o fluxo completo usando os PDFs reais do projeto."""

    def test_extracts_and_validates_three_policies(self):
        """Confirma que as três políticas são extraídas e aprovadas."""

        # Executa a mesma extração usada pelo programa principal.
        payload = extract_all(
            PROJECT_ROOT / "Politicas",
            PROJECT_ROOT / "config" / "policies.json",
        )

        # Confere os resultados gerais esperados para o conjunto de documentos.
        self.assertEqual(payload["summary"]["status"], "approved")
        self.assertEqual(payload["summary"]["document_count"], 3)
        self.assertEqual(payload["summary"]["page_count"], 9)

        # Confere individualmente a validação, o conteúdo e a identidade de cada PDF.
        for document in payload["documents"]:
            self.assertEqual(document["validation"]["status"], "approved")
            self.assertGreater(document["character_count"], 1000)
            self.assertEqual(len(document["source_sha256"]), 64)


# Permite executar os testes diretamente por este arquivo.
if __name__ == "__main__":
    unittest.main()
