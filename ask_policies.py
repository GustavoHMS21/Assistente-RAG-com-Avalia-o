"""Executa o fluxo RAG completo: recuperação, geração e apresentação das fontes."""

# Permite usar recursos modernos de tipagem sem avaliá-los imediatamente.
from __future__ import annotations

# Importa ferramentas para argumentos de terminal, mensagens de erro e caminhos.
import argparse
import sys
from pathlib import Path

# Importa a geração fundamentada e suas configurações iniciais.
from src.answerer import (
    DEFAULT_MIN_RELEVANCE_SCORE,
    DEFAULT_OLLAMA_MODEL,
    create_ollama_client,
    friendly_api_error,
    generate_grounded_answer,
)

# Importa a busca semântica implementada na etapa anterior.
from src.retriever import (
    DEFAULT_TOP_K,
    load_embeddings_json,
    load_retrieval_model,
    search_embeddings,
)


# Descobre automaticamente a pasta principal do projeto.
PROJECT_ROOT = Path(__file__).resolve().parent


def configure_console_encoding() -> None:
    """Configura o terminal para apresentar corretamente os textos em português."""

    # Obtém o método de forma segura para não gerar avisos no Pylance.
    stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
    stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)

    # Aplica UTF-8 somente quando o terminal oferece esse recurso.
    if callable(stdout_reconfigure):
        stdout_reconfigure(encoding="utf-8")
    if callable(stderr_reconfigure):
        stderr_reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Define e lê as opções que podem ser informadas pelo terminal."""

    # Cria o interpretador que organiza as opções do comando.
    parser = argparse.ArgumentParser(description="Pergunte às políticas internas de RH.")

    # Aceita a pergunta no comando ou permite digitá-la interativamente depois.
    parser.add_argument("question", nargs="*", help="Pergunta para o assistente de RH.")

    # Permite experimentar outro modelo sem alterar o código.
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)

    # Define quantos chunks serão entregues ao gerador.
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    # Define o limite abaixo do qual a aplicação recusará localmente.
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_RELEVANCE_SCORE,
    )

    # Permite trocar o arquivo de embeddings sem modificar o código.
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "policy_embeddings.json",
    )

    # Lê o comando digitado e devolve os valores organizados.
    return parser.parse_args()


def main() -> int:
    """Coordena a pergunta, recuperação, chamada da API e fontes."""

    # Prepara o terminal antes de imprimir conteúdo das políticas.
    configure_console_encoding()

    # Obtém as opções e solicita a pergunta quando ela não veio no comando.
    args = parse_args()
    question = " ".join(args.question).strip()
    if not question:
        question = input("Digite sua pergunta sobre as políticas: ").strip()

    # Executa todas as partes do RAG e apresenta erros sem revelar a chave.
    try:
        embeddings_payload = load_embeddings_json(args.input)
        retrieval_model = load_retrieval_model(embeddings_payload)
        results = search_embeddings(
            embeddings_payload,
            question=question,
            model=retrieval_model,
            top_k=args.top_k,
        )
        client = create_ollama_client()
        answer_payload = generate_grounded_answer(
            question,
            results=results,
            client=client,
            model=args.model,
            min_relevance_score=args.min_score,
        )
    except Exception as exc:
        print(f"Erro: {friendly_api_error(exc)}", file=sys.stderr)
        return 1

    # Mostra a resposta final de forma separada dos detalhes técnicos.
    print("\nRESPOSTA")
    print("=" * 72)
    print(answer_payload["answer"])

    # Mostra fontes verificáveis geradas pelo código, não pelo modelo.
    if answer_payload["sources"]:
        print("\nFONTES RECUPERADAS")
        print("=" * 72)
        for source_number, source in enumerate(answer_payload["sources"], start=1):
            print(
                f"Fonte {source_number}: {source['document_title']} | "
                f"página {source['page']} | similaridade {source['score']:.4f} | "
                f"{source['chunk_id']}"
            )

    # Informa o modelo usado sem apresentar credenciais ou dados sensíveis.
    print(f"\nModelo gerador: {answer_payload['model']}")

    # Retorna dois apenas quando a proteção recusou por falta de evidência.
    return 2 if answer_payload["refused"] else 0


# Inicia o programa somente quando este arquivo é executado diretamente.
if __name__ == "__main__":
    raise SystemExit(main())
