"""Leitura e validação das políticas em PDF."""

# Permite usar recursos modernos de tipagem sem avaliá-los imediatamente.
from __future__ import annotations

# Importa as ferramentas da biblioteca padrão usadas no processamento.
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Importa a biblioteca externa responsável por abrir e ler arquivos PDF.
from pypdf import PdfReader


def normalize_page_text(text: str) -> str:
    """Remove espaços acidentais sem destruir a separação entre linhas."""

    # Cria uma lista vazia que receberá somente as linhas úteis e limpas.
    cleaned_lines: list[str] = []

    # Padroniza as quebras de linha e examina o texto uma linha por vez.
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        # Troca vários espaços ou tabulações por um único espaço.
        line = re.sub(r"[ \t]+", " ", raw_line).strip()

        # Guarda apenas linhas que ainda possuem algum conteúdo.
        if line:
            cleaned_lines.append(line)

    # Reúne as linhas limpas novamente em um único texto.
    return "\n".join(cleaned_lines)


def normalized_for_search(text: str) -> str:
    """Cria uma versão comparável, sem diferença entre maiúsculas e acentos."""

    # Converte para minúsculas e separa letras de seus acentos.
    decomposed = unicodedata.normalize("NFKD", text.casefold())

    # Remove os sinais de acentuação que foram separados na etapa anterior.
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))

    # Padroniza todos os espaços para tornar a comparação mais confiável.
    return re.sub(r"\s+", " ", without_accents).strip()


def file_sha256(path: Path) -> str:
    """Calcula uma impressão digital que identifica exatamente o arquivo."""

    # Cria o objeto que calculará o código SHA-256.
    digest = hashlib.sha256()

    # Abre o PDF em modo binário, pois o cálculo considera os bytes do arquivo.
    with path.open("rb") as stream:
        # Lê em blocos de 1 MB para não carregar um arquivo grande inteiro na memória.
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    # Devolve a impressão digital como uma sequência de 64 caracteres.
    return digest.hexdigest()


def load_config(config_path: Path) -> dict[str, dict[str, Any]]:
    """Carrega o arquivo JSON que descreve as políticas esperadas."""

    # Interrompe a execução com uma mensagem clara se a configuração não existir.
    if not config_path.exists():
        raise FileNotFoundError(f"Configuração não encontrada: {config_path}")

    # Abre o JSON como texto UTF-8 e o converte para um dicionário Python.
    with config_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    # Garante que a configuração tenha o formato mínimo necessário.
    if not isinstance(data, dict) or not data:
        raise ValueError("O arquivo de configuração deve conter pelo menos uma política.")

    # Entrega a configuração validada para as próximas etapas.
    return data


def extract_policy(pdf_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Extrai o texto e os dados de uma única política em PDF."""

    # Abre o PDF e cria a lista que armazenará o resultado de cada página.
    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, Any]] = []

    # Percorre todas as páginas; a contagem começa em 1, como no documento real.
    for page_number, page in enumerate(reader.pages, start=1):
        # Extrai e limpa o texto; usa texto vazio caso a página não seja legível.
        text = normalize_page_text(page.extract_text() or "")

        # Guarda o texto da página junto com dados que permitirão citar sua origem.
        pages.append(
            {
                "document_id": metadata["code"],
                "document_title": metadata["title"],
                "file_name": pdf_path.name,
                "version": metadata["version"],
                "effective_date": metadata["effective_date"],
                "category": metadata["category"],
                "page": page_number,
                "character_count": len(text),
                "text": text,
            }
        )

    # Reúne todas as páginas para permitir a validação do documento completo.
    full_text = "\n".join(page["text"] for page in pages)

    # Confere se o documento possui páginas, texto e as frases esperadas.
    validation = validate_policy(pdf_path.name, full_text, metadata, pages)

    # Monta o objeto final dessa política com origem, conteúdo e validação.
    return {
        "file_name": pdf_path.name,
        "document_id": metadata["code"],
        "document_title": metadata["title"],
        "version": metadata["version"],
        "effective_date": metadata["effective_date"],
        "category": metadata["category"],
        "source_sha256": file_sha256(pdf_path),
        "page_count": len(pages),
        "character_count": len(full_text),
        "validation": validation,
        "pages": pages,
    }


def validate_policy(
    file_name: str,
    full_text: str,
    metadata: dict[str, Any],
    pages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verifica se uma política foi lida corretamente e está completa."""

    # Prepara o texto para uma comparação que ignora maiúsculas e acentos.
    searchable_text = normalized_for_search(full_text)

    # Localiza frases obrigatórias da configuração que não apareceram no PDF.
    missing_phrases = [
        phrase
        for phrase in metadata.get("expected_phrases", [])
        if normalized_for_search(phrase) not in searchable_text
    ]

    # Identifica páginas das quais nenhum texto pôde ser extraído.
    empty_pages = [page["page"] for page in pages if not page["text"]]

    # Cria uma lista para reunir todos os problemas encontrados.
    errors: list[str] = []

    # Registra um erro se o arquivo não contiver nenhuma página.
    if not pages:
        errors.append("O PDF não possui páginas.")

    # Registra quais páginas não possuem texto selecionável.
    if empty_pages:
        errors.append(f"Páginas sem texto selecionável: {empty_pages}")

    # Registra quais informações esperadas não foram encontradas.
    if missing_phrases:
        errors.append(f"Frases esperadas ausentes: {missing_phrases}")

    # Aprova o documento somente se a lista de erros estiver vazia.
    return {
        "status": "approved" if not errors else "failed",
        "file_name": file_name,
        "empty_pages": empty_pages,
        "missing_expected_phrases": missing_phrases,
        "errors": errors,
    }


def extract_all(policies_dir: Path, config_path: Path) -> dict[str, Any]:
    """Processa todas as políticas e cria um resultado único."""

    # Carrega a lista de documentos esperados e encontra os PDFs da pasta.
    config = load_config(config_path)
    pdf_files = sorted(policies_dir.glob("*.pdf"))

    # Interrompe o processo se a pasta não tiver nenhum PDF.
    if not pdf_files:
        raise FileNotFoundError(f"Nenhum PDF encontrado em: {policies_dir}")

    # Compara os nomes configurados com os arquivos realmente encontrados.
    configured_names = set(config)
    discovered_names = {path.name for path in pdf_files}
    missing_files = sorted(configured_names - discovered_names)
    unknown_files = sorted(discovered_names - configured_names)

    # Impede a continuação se algum PDF obrigatório estiver ausente.
    if missing_files:
        raise FileNotFoundError(f"PDFs configurados não encontrados: {missing_files}")

    # Impede o uso acidental de um PDF que ainda não possui configuração.
    if unknown_files:
        raise ValueError(f"PDFs sem configuração: {unknown_files}")

    # Extrai cada documento usando os metadados correspondentes ao seu nome.
    documents = [extract_policy(path, config[path.name]) for path in pdf_files]

    # O conjunto completo só é aprovado se todos os documentos forem aprovados.
    approved = all(doc["validation"]["status"] == "approved" for doc in documents)

    # Monta o resultado geral com data, totais e o conteúdo dos documentos.
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "status": "approved" if approved else "failed",
            "document_count": len(documents),
            "page_count": sum(doc["page_count"] for doc in documents),
            "character_count": sum(doc["character_count"] for doc in documents),
        },
        "documents": documents,
    }


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    """Salva o resultado da extração em um arquivo JSON legível."""

    # Cria as pastas de destino automaticamente caso elas ainda não existam.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Grava o JSON preservando acentos e usando indentação para facilitar a leitura.
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
