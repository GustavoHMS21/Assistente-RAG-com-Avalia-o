"""Divide as páginas extraídas dos PDFs em trechos menores e pesquisáveis."""

# Permite usar recursos modernos de tipagem sem avaliá-los imediatamente.
from __future__ import annotations

# Importa ferramentas para JSON, datas, caminhos e tipos genéricos.
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Define os valores iniciais do chunking usados pelo projeto.
DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP_CHARS = 150
DEFAULT_MIN_CHARS = 150


def validate_chunk_settings(max_chars: int, overlap_chars: int, min_chars: int) -> None:
    """Confere se os tamanhos escolhidos podem produzir chunks válidos."""

    # Evita chunks pequenos demais para carregar contexto útil.
    if max_chars < 200:
        raise ValueError("max_chars deve ser igual ou superior a 200.")

    # A sobreposição pode ser zero, mas nunca negativa.
    if overlap_chars < 0:
        raise ValueError("overlap_chars não pode ser negativo.")

    # A parte repetida precisa ser menor que o tamanho total do chunk.
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars deve ser menor que max_chars.")

    # O tamanho mínimo precisa caber dentro do tamanho máximo.
    if min_chars < 1 or min_chars > max_chars:
        raise ValueError("min_chars deve estar entre 1 e max_chars.")


def split_oversized_line(line: str, max_chars: int) -> list[str]:
    """Divide uma linha muito grande sem cortar palavras no meio."""

    # Se a linha já couber em um chunk, nenhuma divisão é necessária.
    if len(line) <= max_chars:
        return [line]

    # Separa a linha em palavras para montar partes dentro do limite escolhido.
    words = line.split()
    parts: list[str] = []
    current_words: list[str] = []

    # Acrescenta palavras enquanto a parte atual continuar dentro do limite.
    for word in words:
        candidate = " ".join([*current_words, word])

        # Fecha a parte atual antes de adicionar uma palavra que excederia o limite.
        if current_words and len(candidate) > max_chars:
            parts.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words.append(word)

    # Guarda a última parte que restou após o término do laço.
    if current_words:
        parts.append(" ".join(current_words))

    # Trata uma sequência excepcional sem espaços, dividindo-a por caracteres.
    safe_parts: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            safe_parts.append(part)
        else:
            safe_parts.extend(part[index : index + max_chars] for index in range(0, len(part), max_chars))

    # Devolve somente partes que respeitam o tamanho máximo.
    return safe_parts


def text_to_segments(text: str, max_chars: int) -> list[str]:
    """Reconstrói linhas quebradas do PDF em blocos lógicos de texto."""

    # Padroniza as quebras de linha produzidas em diferentes sistemas.
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Cria listas para os segmentos prontos e para as linhas do bloco atual.
    segments: list[str] = []
    current_lines: list[str] = []

    # Esta função interna fecha o bloco atual e o adiciona à lista de segmentos.
    def flush_current_lines() -> None:
        if not current_lines:
            return

        # Une as quebras visuais do PDF para reconstruir frases e parágrafos.
        logical_block = " ".join(current_lines)
        segments.extend(split_oversized_line(logical_block, max_chars))
        current_lines.clear()

    # Percorre cada linha e identifica títulos, marcadores e finais de frase.
    for raw_line in normalized_text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()

        # Uma linha vazia encerra o bloco lógico que estava sendo reconstruído.
        if not line:
            flush_current_lines()
            continue

        # Títulos numerados e marcadores iniciam um novo bloco independente.
        is_heading = bool(re.match(r"^\d+(?:\.\d+)*\s+\S", line))
        is_bullet = bool(re.match(r"^[●•▪◦]\s*", line))
        if is_heading or is_bullet:
            flush_current_lines()
            segments.extend(split_oversized_line(line, max_chars))
            continue

        # Linhas comuns são acumuladas para desfazer quebras visuais do PDF.
        current_lines.append(line)

        # Pontuação final indica que o bloco já contém uma ideia completa.
        if re.search(r"[.!?;:]$", line):
            flush_current_lines()

    # Fecha o último bloco caso a página termine sem pontuação.
    flush_current_lines()

    # Entrega unidades lógicas que poderão ser agrupadas sem cortar frases.
    return segments


def joined_length(segments: list[str]) -> int:
    """Calcula o tamanho dos segmentos considerando as quebras entre eles."""

    # O texto final possui uma quebra de linha entre cada par de segmentos.
    return sum(len(segment) for segment in segments) + max(0, len(segments) - 1)


def find_overlap_start(
    segments: list[str],
    current_start: int,
    current_end: int,
    overlap_chars: int,
) -> int:
    """Encontra de onde o próximo chunk deve começar para repetir algum contexto."""

    # Não cria sobreposição quando ela foi desativada ou há apenas um segmento.
    if overlap_chars == 0 or current_end - current_start <= 1:
        return current_end

    # Começa no fim do chunk atual e tenta incluir segmentos anteriores.
    overlap_start = current_end
    overlap_size = 0

    # Caminha para trás sem voltar ao mesmo início, o que evitaria um laço infinito.
    while overlap_start - 1 > current_start:
        previous_segment = segments[overlap_start - 1]
        separator_size = 1 if overlap_size else 0
        candidate_size = overlap_size + separator_size + len(previous_segment)

        # Quando o primeiro bloco já supera a meta, repete o bloco inteiro.
        # Preservar uma frase completa é mais importante que cortar no número exato.
        if candidate_size > overlap_chars:
            if overlap_start == current_end:
                overlap_start -= 1
            break

        overlap_start -= 1
        overlap_size = candidate_size

    # Se nenhum segmento coube, o próximo chunk começa após o atual.
    return overlap_start if overlap_start < current_end else current_end


def chunk_page_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[dict[str, Any]]:
    """Divide o texto de uma página e devolve chunks com informações de tamanho."""

    # Garante que as configurações recebidas sejam coerentes.
    validate_chunk_settings(max_chars, overlap_chars, min_chars)

    # Converte as linhas da página em unidades que não ultrapassam o limite.
    segments = text_to_segments(text, max_chars)

    # Uma página sem texto não produz chunks vazios.
    if not segments:
        return []

    # Prepara a lista final e o índice do primeiro segmento ainda não processado.
    chunks: list[dict[str, Any]] = []
    start = 0

    # Continua até que todos os segmentos da página tenham sido incluídos.
    while start < len(segments):
        end = start
        selected_segments: list[str] = []

        # Acrescenta segmentos enquanto o chunk permanecer dentro do limite.
        while end < len(segments):
            candidate_segments = [*selected_segments, segments[end]]
            if selected_segments and joined_length(candidate_segments) > max_chars:
                break
            selected_segments = candidate_segments
            end += 1

        # Reúne os segmentos preservando quebras úteis para leitura humana.
        chunk_text = "\n".join(selected_segments)

        # Registra o chunk e sinaliza quando ele ficou abaixo do tamanho desejado.
        chunks.append(
            {
                "text": chunk_text,
                "character_count": len(chunk_text),
                "is_short": len(chunk_text) < min_chars,
            }
        )

        # Encerra a página quando o último segmento já foi incluído.
        # Isso evita criar um chunk adicional contendo somente texto repetido.
        if end >= len(segments):
            break

        # Calcula o próximo início, incluindo uma pequena repetição quando possível.
        next_start = find_overlap_start(segments, start, end, overlap_chars)

        # Garante avanço mesmo em casos incomuns de configuração ou texto.
        start = next_start if next_start > start else end

    # Entrega todos os chunks produzidos para a página.
    return chunks


def build_chunks(
    extracted_payload: dict[str, Any],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> dict[str, Any]:
    """Cria chunks de todos os documentos presentes no JSON de extração."""

    # Valida as configurações antes de iniciar o processamento completo.
    validate_chunk_settings(max_chars, overlap_chars, min_chars)

    # Confere se o arquivo de entrada contém uma lista de documentos.
    documents = extracted_payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("O arquivo de entrada não contém documentos para processar.")

    # Prepara a lista geral e os contadores usados no resumo.
    all_chunks: list[dict[str, Any]] = []
    processed_pages = 0

    # Percorre cada política extraída anteriormente.
    for document in documents:
        document_chunk_number = 0
        pages = document.get("pages", [])

        # Percorre as páginas mantendo a referência exata da origem.
        for page in pages:
            page_number = int(page["page"])
            page_chunks = chunk_page_text(
                page.get("text", ""),
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                min_chars=min_chars,
            )

            # Conta somente páginas que realmente produziram algum conteúdo.
            if page_chunks:
                processed_pages += 1

            # Enriquece cada trecho com dados do documento e da página.
            for page_chunk_number, page_chunk in enumerate(page_chunks, start=1):
                document_chunk_number += 1
                chunk_id = (
                    f"{document['document_id']}-P{page_number:03d}-C{page_chunk_number:03d}"
                )

                # Salva metadados suficientes para busca, filtros e citação da fonte.
                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": document["document_id"],
                        "document_title": document["document_title"],
                        "file_name": document["file_name"],
                        "version": document["version"],
                        "effective_date": document["effective_date"],
                        "category": document["category"],
                        "page": page_number,
                        "page_chunk_number": page_chunk_number,
                        "document_chunk_number": document_chunk_number,
                        "character_count": page_chunk["character_count"],
                        "is_short": page_chunk["is_short"],
                        "text": page_chunk["text"],
                    }
                )

    # Impede a geração silenciosa de um arquivo sem nenhum trecho pesquisável.
    if not all_chunks:
        raise ValueError("Nenhum chunk foi produzido a partir dos documentos.")

    # Calcula números úteis para avaliar rapidamente o resultado do chunking.
    character_counts = [chunk["character_count"] for chunk in all_chunks]
    short_chunk_count = sum(1 for chunk in all_chunks if chunk["is_short"])

    # Monta o arquivo final com configuração, resumo e chunks.
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_schema_version": extracted_payload.get("schema_version"),
        "chunking_config": {
            "strategy": "page_aware_logical_blocks",
            "max_chars": max_chars,
            "overlap_chars": overlap_chars,
            "min_chars": min_chars,
        },
        "summary": {
            "document_count": len(documents),
            "page_count": processed_pages,
            "chunk_count": len(all_chunks),
            "short_chunk_count": short_chunk_count,
            "minimum_chunk_chars": min(character_counts),
            "maximum_chunk_chars": max(character_counts),
            "average_chunk_chars": round(sum(character_counts) / len(character_counts), 2),
        },
        "chunks": all_chunks,
    }


def load_extracted_json(input_path: Path) -> dict[str, Any]:
    """Carrega o JSON criado na etapa de extração dos PDFs."""

    # Apresenta um erro direto quando a etapa anterior ainda não foi executada.
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")

    # Abre o arquivo preservando os caracteres da língua portuguesa.
    with input_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    # Garante que a raiz do JSON tenha o formato de objeto esperado.
    if not isinstance(payload, dict):
        raise ValueError("O arquivo de entrada deve conter um objeto JSON.")

    # Devolve o conteúdo pronto para a criação dos chunks.
    return payload


def save_chunks_json(payload: dict[str, Any], output_path: Path) -> None:
    """Salva os chunks em um JSON formatado e fácil de inspecionar."""

    # Cria automaticamente a pasta de destino quando necessário.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Grava o resultado com acentos preservados e indentação legível.
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
