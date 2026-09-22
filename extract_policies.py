"""Executa a primeira etapa do ProcedIA: leitura e validação dos PDFs."""

# Permite usar recursos modernos de tipagem sem avaliá-los imediatamente.
from __future__ import annotations

# Importa as ferramentas da biblioteca padrão usadas neste arquivo.
import argparse
import sys
from pathlib import Path

# Importa do nosso projeto as funções que leem os PDFs e salvam o resultado.
from src.pdf_reader import extract_all, save_json


# Descobre automaticamente a pasta principal do projeto.
# Assim, o programa funciona mesmo quando for iniciado por outro diretório.
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Define e lê as opções que podem ser passadas pelo terminal."""

    # Cria o interpretador responsável por entender os argumentos do comando.
    parser = argparse.ArgumentParser(description="Extrai as políticas em PDF página por página.")

    # Permite trocar a pasta dos PDFs; se nada for informado, usa Politicas/.
    parser.add_argument(
        "--policies-dir",
        type=Path,
        default=PROJECT_ROOT / "Politicas",
        help="Pasta que contém os PDFs.",
    )

    # Permite trocar o arquivo que descreve e valida as políticas esperadas.
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "policies.json",
        help="Arquivo de metadados e validações.",
    )

    # Permite escolher onde o JSON resultante será gravado.
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "policy_pages.json",
        help="Arquivo JSON de saída.",
    )

    # Lê o que foi digitado no terminal e devolve os valores organizados.
    return parser.parse_args()


def main() -> int:
    """Coordena a extração e devolve um código que informa se ela funcionou."""

    # Obtém os caminhos definidos pelo usuário ou os caminhos padrão do projeto.
    args = parse_args()

    # Executa a leitura e salva o resultado; qualquer erro é apresentado claramente.
    try:
        payload = extract_all(args.policies_dir, args.config)
        save_json(payload, args.output)
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    # Obtém apenas o resumo para mostrar uma resposta curta no terminal.
    summary = payload["summary"]

    # Exibe os principais números da execução para facilitar a conferência.
    print("Extração concluída")
    print(f"Status: {summary['status']}")
    print(f"Documentos: {summary['document_count']}")
    print(f"Páginas: {summary['page_count']}")
    print(f"Caracteres extraídos: {summary['character_count']}")
    print(f"Arquivo gerado: {args.output.resolve()}")

    # Retorna 0 quando tudo foi aprovado ou 2 quando alguma validação falhou.
    return 0 if summary["status"] == "approved" else 2


# Inicia o programa somente quando este arquivo é executado diretamente.
# Se ele for importado por outro arquivo, a função main não roda automaticamente.
if __name__ == "__main__":
    raise SystemExit(main())
