"""Testes automáticos do ranking e da avaliação da busca semântica."""

# Importa a ferramenta usada para criar testes automáticos.
import unittest

# Importa as funções de busca e avaliação que serão verificadas.
from src.retriever import evaluate_retrieval, search_embeddings


class FakeQueryModel:
    """Simula embeddings de perguntas sem carregar o modelo real."""

    def encode(self, texts, **kwargs):
        """Converte perguntas conhecidas em vetores normalizados previsíveis."""

        # Define uma direção vetorial para cada assunto do teste.
        vectors_by_question = {
            "pergunta sobre férias": [1.0, 0.0, 0.0],
            "pergunta sobre remoto": [0.0, 1.0, 0.0],
            "pergunta sobre privacidade": [0.0, 0.0, 1.0],
        }

        # Devolve um vetor para cada texto recebido pelo método.
        return [vectors_by_question[text] for text in texts]


def create_fake_embeddings_payload():
    """Cria três documentos pequenos, cada um apontando para um eixo diferente."""

    # Monta um arquivo mínimo com a mesma estrutura usada pelo projeto real.
    return {
        "embedding_config": {"dimension": 3, "model_name": "modelo-de-teste"},
        "embeddings": [
            {
                "chunk_id": "FERIAS-001",
                "document_id": "FERIAS",
                "document_title": "Política de Férias",
                "file_name": "ferias.pdf",
                "category": "Férias",
                "page": 1,
                "text": "É possível vender parte das férias.",
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "chunk_id": "REMOTO-001",
                "document_id": "REMOTO",
                "document_title": "Política de Trabalho Remoto",
                "file_name": "remoto.pdf",
                "category": "Trabalho remoto",
                "page": 1,
                "text": "O trabalho remoto depende de aprovação.",
                "embedding": [0.0, 1.0, 0.0],
            },
            {
                "chunk_id": "PRIVACIDADE-001",
                "document_id": "PRIVACIDADE",
                "document_title": "Política de Privacidade",
                "file_name": "privacidade.pdf",
                "category": "Privacidade",
                "page": 1,
                "text": "O acesso aos dados pessoais é restrito.",
                "embedding": [0.0, 0.0, 1.0],
            },
        ],
    }


class SemanticSearchTests(unittest.TestCase):
    """Verifica se o ranking coloca o assunto correto em primeiro lugar."""

    def test_ranks_most_similar_chunk_first(self):
        """Confirma a ordenação pela pontuação de similaridade."""

        # Executa uma busca sobre trabalho remoto no conjunto simulado.
        results = search_embeddings(
            create_fake_embeddings_payload(),
            question="pergunta sobre remoto",
            model=FakeQueryModel(),
            top_k=3,
        )

        # O vetor idêntico precisa ocupar a primeira posição com pontuação máxima.
        self.assertEqual(results[0]["document_id"], "REMOTO")
        self.assertEqual(results[0]["rank"], 1)
        self.assertAlmostEqual(results[0]["score"], 1.0)

    def test_applies_minimum_score(self):
        """Confirma que resultados fracos podem ser removidos por um limite."""

        # Com limite alto, somente o documento idêntico deve permanecer.
        results = search_embeddings(
            create_fake_embeddings_payload(),
            question="pergunta sobre férias",
            model=FakeQueryModel(),
            top_k=3,
            min_score=0.9,
        )

        # Verifica quantidade e identidade do único resultado restante.
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["document_id"], "FERIAS")


class RetrievalEvaluationTests(unittest.TestCase):
    """Verifica o cálculo da métrica Hit@K."""

    def test_calculates_perfect_hit_at_one(self):
        """Confirma 100% quando todas as políticas corretas ficam em primeiro."""

        # Define três perguntas e seus documentos esperados.
        questions = [
            {
                "id": "Q1",
                "question": "pergunta sobre férias",
                "expected_document_id": "FERIAS",
            },
            {
                "id": "Q2",
                "question": "pergunta sobre remoto",
                "expected_document_id": "REMOTO",
            },
            {
                "id": "Q3",
                "question": "pergunta sobre privacidade",
                "expected_document_id": "PRIVACIDADE",
            },
        ]

        # Executa a avaliação considerando somente o primeiro resultado.
        evaluation = evaluate_retrieval(
            create_fake_embeddings_payload(),
            questions=questions,
            model=FakeQueryModel(),
            top_k=1,
        )

        # Todas as perguntas devem encontrar seu documento correto.
        self.assertEqual(evaluation["summary"]["hit_count"], 3)
        self.assertEqual(evaluation["summary"]["miss_count"], 0)
        self.assertEqual(evaluation["summary"]["hit_at_k"], 1.0)


# Permite executar os testes diretamente por este arquivo.
if __name__ == "__main__":
    unittest.main()
