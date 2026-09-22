"""Avalia se a busca encontra a política correta para perguntas conhecidas."""

# Permite usar recursos modernos de tipagem sem avaliá-los imediatamente.
from __future__ import annotations

# Importa ferramentas para argumentos de terminal, mensagens de erro e caminhos.
import argparse
import sys
from pathlib import Path

# Importa as funções responsáveis pela avaliação da recuperação semântica.
from src.retriever import (
    DEFAULT_TOP_K,
    evaluate_retrieval,
    load_embeddings_json,
    load_evaluation_questions,
    load_retrieval_model,
    save_evaluation_json,
)


# Descobre automaticamente a pasta principal do projeto.
PROJECT_ROOT = Path(__file__).resolve().parent


def configure_console_encoding() -> None:
    """Configura o terminal para apresentar corretamente os textos em português."""

    # Obtém o método de forma segura, pois nem todo tipo de terminal oferece reconfigure.
    stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
    stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)

    # Substitui a codificação limitada usada por algumas sessões do Windows.
    if callable(stdout_reconfigure):
        stdout_reconfigure(encoding="utf-8")
    if callable(stderr_reconfigure):
        stderr_reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Define e lê os caminhos e valores usados pela avaliação."""

    # Cria o interpretador que organiza as opções do comando.
    parser = argparse.ArgumentParser(description="Avalia a qualidade da busca semântica.")

    # Define o arquivo que contém os embeddings pesquisáveis.
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "policy_embeddings.json",
    )

    # Define o conjunto de perguntas com respostas esperadas.
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "config" / "evaluation_questions.json",
    )

    # Define onde o relatório detalhado será armazenado.
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "retrieval_evaluation.json",
    )

    # Permite mudar quantos resultados serão considerados em cada acerto.
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    # Lê o comando digitado e devolve os valores organizados.
    return parser.parse_args()


def main() -> int:
    """Executa os casos conhecidos e apresenta a métrica Hit@K."""

    # Prepara o terminal antes de imprimir perguntas com acentos.
    configure_console_encoding()

    # Obtém as opções informadas pelo usuário.
    args = parse_args()

    # Carrega dados, modelo e perguntas antes de calcular a avaliação.
    try:
        embeddings_payload = load_embeddings_json(args.embeddings)
        questions = load_evaluation_questions(args.questions)
        model = load_retrieval_model(embeddings_payload)
        evaluation = evaluate_retrieval(
            embeddings_payload,
            questions=questions,
            model=model,
            top_k=args.top_k,
        )
        save_evaluation_json(evaluation, args.output)
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    # Mostra cada pergunta e a posição em que apareceu a política correta.
    for result in evaluation["results"]:
        status = "ACERTOU" if result["hit"] else "ERROU"
        rank = result["expected_document_rank"] or "não encontrada"
        print(f"[{status}] {result['question']}")
        print(f"  Política correta na posição: {rank}")

    # Apresenta o resumo final em número absoluto e percentual.
    summary = evaluation["summary"]
    print("\nAvaliação concluída")
    print(f"Métrica: {evaluation['metric']}")
    print(f"Acertos: {summary['hit_count']}/{summary['question_count']}")
    print(f"Resultado: {summary['hit_at_k']:.1%}")
    print(f"Relatório: {args.output.resolve()}")

    # Retorna sucesso somente quando todas as perguntas forem recuperadas.
    return 0 if summary["miss_count"] == 0 else 2


# Inicia o programa somente quando este arquivo é executado diretamente.
if __name__ == "__main__":
    raise SystemExit(main())
