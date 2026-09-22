"""Testes automáticos da geração fundamentada e de suas proteções."""

# Importa ferramentas para criar objetos simulados e testes automáticos.
from types import SimpleNamespace
import unittest

# Importa as funções da camada de resposta que serão verificadas.
from src.answerer import build_grounded_context, friendly_api_error, generate_grounded_answer


class FakeOllamaClient:
    """Simula o cliente do Ollama sem realizar chamadas externas."""

    def __init__(self):
        """Prepara o contador e o local onde os argumentos serão guardados."""

        # Esses campos permitem confirmar se a chamada simulada aconteceu corretamente.
        self.call_count = 0
        self.last_arguments = None

    def chat(self, **kwargs):
        """Guarda a chamada e devolve uma resposta previsível."""

        # Registra os argumentos sem depender de um servidor Ollama real.
        self.call_count += 1
        self.last_arguments = kwargs

        # Imita apenas os campos do SDK utilizados pelo projeto.
        return SimpleNamespace(
            message=SimpleNamespace(content="É possível converter até 10 dias em abono [Fonte 1]."),
        )


def create_search_results(score: float = 0.75):
    """Cria resultados mínimos com a estrutura produzida pelo recuperador."""

    # Devolve uma evidência fictícia e rastreável para os testes.
    return [
        {
            "chunk_id": "POL-RH-003-P001-C003",
            "document_id": "POL-RH-003",
            "document_title": "Política de Férias Ausências e Licenças",
            "file_name": "politica_ferias_ausencias_licencas.pdf",
            "category": "Férias",
            "page": 1,
            "score": score,
            "text": "O colaborador poderá converter até 10 dias em abono pecuniário.",
        }
    ]


class GroundedAnswerTests(unittest.TestCase):
    """Verifica contexto, chamada ao Ollama, fontes e recusas."""

    def test_builds_context_with_traceable_source(self):
        """Confirma que o prompt contém conteúdo e identificação da origem."""

        # Monta o contexto usando a mesma evidência recebida da busca.
        context = build_grounded_context(create_search_results())

        # Confere os dados necessários para fundamentação e citação.
        self.assertIn("[Fonte 1]", context)
        self.assertIn("Página: 1", context)
        self.assertIn("converter até 10 dias", context)

    def test_calls_ollama_with_grounded_context(self):
        """Confirma modelo, evidência, configuração de saída e fontes locais."""

        # Executa a geração com o cliente simulado.
        client = FakeOllamaClient()
        payload = generate_grounded_answer(
            "Quantos dias posso vender?",
            results=create_search_results(),
            client=client,
            model="modelo-de-teste",
        )

        # Confere o resultado mostrado ao usuário.
        self.assertFalse(payload["refused"])
        self.assertIn("10 dias", payload["answer"])
        self.assertEqual(payload["sources"][0]["page"], 1)

        # Confere que a chamada usa o contexto recuperado, limita a saída e desativa o thinking.
        arguments = client.last_arguments
        self.assertEqual(client.call_count, 1)
        self.assertEqual(arguments["options"]["num_predict"], 500)
        self.assertFalse(arguments["think"])
        self.assertIn("EVIDÊNCIAS RECUPERADAS", arguments["messages"][-1]["content"])

    def test_refuses_locally_when_evidence_is_weak(self):
        """Confirma que baixa relevância não gera custo nem resposta inventada."""

        # Usa uma pontuação abaixo do limite inicial de 0.35.
        client = FakeOllamaClient()
        payload = generate_grounded_answer(
            "Qual é o cardápio do almoço?",
            results=create_search_results(score=0.20),
            client=client,
        )

        # A aplicação deve recusar sem fazer nenhuma chamada ao SDK.
        self.assertTrue(payload["refused"])
        self.assertIn("Não encontrei informação suficiente", payload["answer"])
        self.assertEqual(client.call_count, 0)

    def test_translates_connection_error(self):
        """Confirma que a indisponibilidade do servidor local gera uma orientação clara."""

        # Simula o erro típico quando o Ollama não está em execução.
        error = ConnectionError("Connection refused")

        # A mensagem deve orientar a abrir o Ollama sem repetir toda a resposta técnica.
        message = friendly_api_error(error)
        self.assertIn("Ollama local", message)
        self.assertNotIn("Connection refused", message)


# Permite executar os testes diretamente por este arquivo.
if __name__ == "__main__":
    unittest.main()
