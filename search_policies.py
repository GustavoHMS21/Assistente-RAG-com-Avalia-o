"""Executa uma busca semântica nas políticas pelo terminal."""

# Permite usar recursos modernos de tipagem sem avaliá-los imediatamente.
from __future__ import annotations

# Importa ferramentas para argumentos de terminal, mensagens de erro e caminhos.
import argparse
import sys
from pathlib import Path

# Importa as funções responsáveis por carregar e pesquisar os embeddings.
from src.retriever import (
    DEFAULT_TOP_K,
    load_embeddings_json,
    load_retrieval_model,
    search_embeddings,
)


# Descobre automaticamente a pasta principal do projeto.
PROJECT_ROOT = Path(__file__).resolve().parent


def configure_console_encoding() -> None:
    """Configura o terminal para exibir acentos e marcadores em UTF-8."""

    # Obtém o método de forma segura, pois nem todo tipo de terminal oferece reconfigure.
    stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
    stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)

    # Terminais modernos do Windows aceitam UTF-8, mas o Python pode iniciar em cp1252.
    if callable(stdout_reconfigure):
        stdout_reconfigure(encoding="utf-8")
    if callable(stderr_reconfigure):
        stderr_reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Define e lê as opções que podem ser informadas pelo terminal."""

    # Cria o interpretador que organiza as opções do comando.
    parser = argparse.ArgumentParser(description="Pesquisa informações nas políticas internas.")

    # Aceita uma pergunta no comando; quando ausente, o programa perguntará depois.
    parser.add_argument("question", nargs="*", help="Pergunta que será pesquisada.")

    # Permite escolher quantos chunks serão apresentados.
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    # Permite ocultar resultados abaixo de uma pontuação escolhida.
    parser.add_argument("--min-score", type=float, default=None)

    # Permite trocar o arquivo de embeddings sem modificar o código.
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "policy_embeddings.json",
    )

    # Lê o comando digitado e devolve os valores organizados.
    return parser.parse_args()


def main() -> int:
    """Recebe a pergunta, executa a busca e mostra resultados com suas fontes."""

    # Prepara o terminal antes de imprimir textos extraídos dos PDFs.
    configure_console_encoding()

    # Obtém as opções informadas pelo usuário.
    args = parse_args()

    # Une as palavras recebidas ou solicita uma pergunta de forma interativa.
    question = " ".join(args.question).strip()
    if not question:
        question = input("Digite sua pergunta sobre as políticas: ").strip()

    # Carrega os embeddings, o mesmo modelo e executa a pesquisa.
    try:
        embeddings_payload = load_embeddings_json(args.input)
        model = load_retrieval_model(embeddings_payload)
        results = search_embeddings(
            embeddings_payload,
            question=question,
            model=model,
            top_k=args.top_k,
            min_score=args.min_score,
        )
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    # Mostra a pergunta para deixar a saída fácil de revisar.
    print(f"\nPergunta: {question}")

    # Informa quando nenhum trecho atingiu o limite opcional.
    if not results:
        print("Nenhum trecho atingiu a pontuação mínima escolhida.")
        return 2

    # Apresenta cada trecho com posição, pontuação e origem verificável.
    for result in results:
        print("\n" + "=" * 72)
        print(f"{result['rank']}º resultado | Similaridade: {result['score']:.4f}")
        print(f"Documento: {result['document_title']}")
        print(f"Página: {result['page']} | Chunk: {result['chunk_id']}")
        print("-" * 72)
        print(result["text"])

    # Retorna zero para informar que a pesquisa foi concluída corretamente.
    return 0


# Inicia o programa somente quando este arquivo é executado diretamente.
if __name__ == "__main__":
    raise SystemExit(main())
