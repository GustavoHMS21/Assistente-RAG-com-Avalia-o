"""Transforma chunks de texto em vetores numéricos para busca semântica."""

# Permite usar recursos modernos de tipagem sem avaliá-los imediatamente.
from __future__ import annotations

# Importa ferramentas para JSON, cálculos, datas, caminhos e tipos genéricos.
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Define o modelo multilíngue usado inicialmente pelo projeto.
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Define quantos textos serão processados juntos em cada lote.
DEFAULT_BATCH_SIZE = 16


def prepare_embedding_text(chunk: dict[str, Any]) -> str:
    """Combina metadados úteis e conteúdo no texto enviado ao modelo."""

    # Obtém os campos principais e remove espaços desnecessários das extremidades.
    title = str(chunk.get("document_title", "")).strip()
    category = str(chunk.get("category", "")).strip()
    text = str(chunk.get("text", "")).strip()

    # Impede a criação de um vetor para um chunk sem conteúdo real.
    if not text:
        raise ValueError(f"O chunk {chunk.get('chunk_id', '<sem id>')} não possui texto.")

    # Acrescenta título e categoria para dar mais contexto semântico ao trecho.
    return f"Documento: {title}\nCategoria: {category}\nConteúdo: {text}"


def load_sentence_transformer(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Carrega o modelo local; na primeira execução ele é baixado automaticamente."""

    # Importa a biblioteca somente quando o modelo realmente for necessário.
    # Isso permite testar a lógica do projeto sem baixar o modelo durante os testes.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "A biblioteca sentence-transformers não está instalada. "
            "Execute: python -m pip install -r requirements.txt"
        ) from exc

    # Carrega do cache local ou baixa o modelo quando ele ainda não estiver disponível.
    return SentenceTransformer(model_name)


def vectors_to_lists(vectors: Any) -> list[list[float]]:
    """Converte a resposta do modelo para listas simples compatíveis com JSON."""

    # Arrays NumPy possuem tolist; modelos falsos de teste já devolvem listas.
    raw_vectors = vectors.tolist() if hasattr(vectors, "tolist") else vectors

    # Converte cada valor para float comum, evitando tipos específicos de bibliotecas.
    return [[float(value) for value in vector] for vector in raw_vectors]


def validate_vectors(vectors: list[list[float]], expected_count: int) -> int:
    """Confere quantidade, dimensão, números válidos e normalização dos vetores."""

    # A quantidade de vetores precisa ser igual à quantidade de chunks enviados.
    if len(vectors) != expected_count:
        raise ValueError(
            f"O modelo devolveu {len(vectors)} vetores para {expected_count} chunks."
        )

    # Um resultado vazio não pode ser usado pela busca semântica.
    if not vectors or not vectors[0]:
        raise ValueError("O modelo não produziu vetores válidos.")

    # Usa o primeiro vetor como referência para a dimensão esperada.
    dimension = len(vectors[0])

    # Verifica se todos os vetores possuem a mesma estrutura e valores utilizáveis.
    for index, vector in enumerate(vectors, start=1):
        if len(vector) != dimension:
            raise ValueError(f"O vetor {index} possui dimensão diferente das demais.")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError(f"O vetor {index} contém um número inválido.")

        # Vetores normalizados devem ter comprimento matemático próximo de 1.
        vector_norm = math.sqrt(sum(value * value for value in vector))
        if not math.isclose(vector_norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(f"O vetor {index} não está normalizado: norma {vector_norm:.6f}.")

    # Devolve a dimensão confirmada para incluí-la no resultado final.
    return dimension


def create_embeddings_payload(
    chunks_payload: dict[str, Any],
    model: Any,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Gera os vetores e combina cada um com os metadados de seu chunk."""

    # Confere se o tamanho do lote informado é válido.
    if batch_size < 1:
        raise ValueError("batch_size deve ser igual ou superior a 1.")

    # Obtém a lista de chunks criada na etapa anterior.
    chunks = chunks_payload.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("O arquivo de entrada não contém chunks para processar.")

    # Prepara os textos enriquecidos que serão enviados ao modelo em um único lote lógico.
    embedding_texts = [prepare_embedding_text(chunk) for chunk in chunks]

    # Solicita vetores normalizados, que permitem usar produto escalar como similaridade.
    raw_vectors = model.encode(
        embedding_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Converte e valida a saída antes de salvar qualquer arquivo.
    vectors = vectors_to_lists(raw_vectors)
    dimension = validate_vectors(vectors, expected_count=len(chunks))

    # Cria a lista que reunirá texto, origem e vetor de cada chunk.
    embedding_records: list[dict[str, Any]] = []

    # Mantém a mesma ordem do JSON de chunks para facilitar auditoria e depuração.
    for chunk, embedding_text, vector in zip(chunks, embedding_texts, vectors, strict=True):
        embedding_records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "document_title": chunk["document_title"],
                "file_name": chunk["file_name"],
                "category": chunk["category"],
                "page": chunk["page"],
                "text": chunk["text"],
                "embedding_text": embedding_text,
                "embedding": vector,
            }
        )

    # Monta o arquivo final com informações necessárias para reproduzir os vetores.
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_schema_version": chunks_payload.get("schema_version"),
        "embedding_config": {
            "provider": "sentence-transformers",
            "model_name": model_name,
            "dimension": dimension,
            "normalized": True,
            "batch_size": batch_size,
        },
        "summary": {
            "document_count": len({record["document_id"] for record in embedding_records}),
            "chunk_count": len(chunks),
            "embedding_count": len(embedding_records),
            "dimension": dimension,
        },
        "embeddings": embedding_records,
    }


def load_chunks_json(input_path: Path) -> dict[str, Any]:
    """Carrega o JSON de chunks produzido na etapa anterior."""

    # Informa claramente quando o chunking ainda não foi executado.
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de chunks não encontrado: {input_path}")

    # Lê o JSON preservando corretamente os caracteres em português.
    with input_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    # Garante que a raiz do arquivo possua o formato esperado.
    if not isinstance(payload, dict):
        raise ValueError("O arquivo de chunks deve conter um objeto JSON.")

    # Devolve o conteúdo pronto para ser transformado em embeddings.
    return payload


def save_embeddings_json(payload: dict[str, Any], output_path: Path) -> None:
    """Salva os embeddings e seus metadados em um arquivo JSON."""

    # Cria a pasta de destino automaticamente quando necessário.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Grava os vetores com acentos preservados e estrutura legível.
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
