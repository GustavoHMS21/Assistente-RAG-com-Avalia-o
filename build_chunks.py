"""Executa a segunda etapa do ProcedIA: criação dos chunks pesquisáveis."""

# Permite usar recursos modernos de tipagem sem avaliá-los imediatamente.
from __future__ import annotations

# Importa ferramentas para argumentos de terminal, mensagens de erro e caminhos.
import argparse
import sys
from pathlib import Path

# Importa as configurações e funções responsáveis pelo chunking.
from src.chunker import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    DEFAULT_OVERLAP_CHARS,
    build_chunks,
    load_extracted_json,
    save_chunks_json,
)


# Descobre automaticamente a pasta principal do projeto.
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Define e lê as opções que podem ser informadas pelo terminal."""

    # Cria o interpretador que organiza as opções do comando.
    parser = argparse.ArgumentParser(description="Divide as políticas extraídas em chunks.")

    # Permite trocar o JSON de entrada sem modificar o código.
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "policy_pages.json",
        help="JSON criado pela etapa de extração.",
    )

    # Permite escolher onde o arquivo de chunks será salvo.
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "policy_chunks.json",
        help="Arquivo JSON de saída.",
    )

    # Define o tamanho máximo aproximado de cada trecho.
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)

    # Define quanto contexto poderá ser repetido entre trechos vizinhos.
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)

    # Define o tamanho abaixo do qual um trecho será sinalizado como curto.
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)

    # Lê o comando digitado e devolve os valores organizados.
    return parser.parse_args()


def main() -> int:
    """Coordena a leitura, criação e gravação dos chunks."""

    # Obtém os caminhos e tamanhos escolhidos pelo usuário.
    args = parse_args()

    # Executa o processo e transforma qualquer problema em uma mensagem clara.
    try:
        extracted_payload = load_extracted_json(args.input)
        chunks_payload = build_chunks(
            extracted_payload,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
            min_chars=args.min_chars,
        )
        save_chunks_json(chunks_payload, args.output)
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    # Obtém o resumo para apresentar somente as informações principais no terminal.
    summary = chunks_payload["summary"]

    # Mostra os números necessários para conferir a execução.
    print("Chunking concluído")
    print(f"Documentos: {summary['document_count']}")
    print(f"Páginas processadas: {summary['page_count']}")
    print(f"Chunks criados: {summary['chunk_count']}")
    print(f"Chunks curtos: {summary['short_chunk_count']}")
    print(f"Menor chunk: {summary['minimum_chunk_chars']} caracteres")
    print(f"Maior chunk: {summary['maximum_chunk_chars']} caracteres")
    print(f"Média: {summary['average_chunk_chars']} caracteres")
    print(f"Arquivo gerado: {args.output.resolve()}")

    # Retorna zero para informar ao sistema que a etapa terminou corretamente.
    return 0


# Inicia o programa somente quando este arquivo é executado diretamente.
if __name__ == "__main__":
    raise SystemExit(main())
