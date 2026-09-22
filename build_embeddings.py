"""Executa a terceira etapa do ProcedIA: geração local dos embeddings."""

# Permite usar recursos modernos de tipagem sem avaliá-los imediatamente.
from __future__ import annotations

# Importa ferramentas para argumentos de terminal, mensagens de erro e caminhos.
import argparse
import sys
from pathlib import Path

# Importa as configurações e funções responsáveis pelos embeddings.
from src.embedder import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    create_embeddings_payload,
    load_chunks_json,
    load_sentence_transformer,
    save_embeddings_json,
)


# Descobre automaticamente a pasta principal do projeto.
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Define e lê as opções que podem ser informadas pelo terminal."""

    # Cria o interpretador que organiza as opções do comando.
    parser = argparse.ArgumentParser(description="Gera embeddings locais para os chunks.")

    # Permite trocar o JSON de entrada sem modificar o código.
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "policy_chunks.json",
        help="JSON criado pela etapa de chunking.",
    )

    # Permite escolher onde os embeddings serão armazenados.
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "policy_embeddings.json",
        help="Arquivo JSON de saída.",
    )

    # Permite substituir o modelo local em experimentos futuros.
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)

    # Permite ajustar quantos chunks serão processados juntos.
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    # Lê o comando digitado e devolve os valores organizados.
    return parser.parse_args()


def main() -> int:
    """Coordena o carregamento do modelo, a geração e a gravação dos vetores."""

    # Obtém os caminhos e configurações escolhidos pelo usuário.
    args = parse_args()

    # Executa o processo e transforma qualquer problema em uma mensagem clara.
    try:
        print(f"Carregando o modelo: {args.model}")
        print("Na primeira execução, o download pode levar alguns minutos.")
        chunks_payload = load_chunks_json(args.input)
        model = load_sentence_transformer(args.model)
        embeddings_payload = create_embeddings_payload(
            chunks_payload,
            model=model,
            model_name=args.model,
            batch_size=args.batch_size,
            show_progress=True,
        )
        save_embeddings_json(embeddings_payload, args.output)
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    # Obtém o resumo para apresentar somente as informações principais no terminal.
    summary = embeddings_payload["summary"]

    # Mostra os números necessários para conferir a execução.
    print("Embeddings gerados com sucesso")
    print(f"Documentos: {summary['document_count']}")
    print(f"Chunks: {summary['chunk_count']}")
    print(f"Vetores: {summary['embedding_count']}")
    print(f"Dimensões por vetor: {summary['dimension']}")
    print(f"Arquivo gerado: {args.output.resolve()}")

    # Retorna zero para informar ao sistema que a etapa terminou corretamente.
    return 0


# Inicia o programa somente quando este arquivo é executado diretamente.
if __name__ == "__main__":
    raise SystemExit(main())
