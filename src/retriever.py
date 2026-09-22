"""Pesquisa os chunks mais próximos de uma pergunta usando similaridade vetorial."""

# Permite usar recursos modernos de tipagem sem avaliá-los imediatamente.
from __future__ import annotations

# Importa ferramentas para JSON, cálculos, datas, caminhos e tipos genéricos.
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reutiliza a carga do modelo e as validações criadas na etapa de embeddings.
from src.embedder import load_sentence_transformer, validate_vectors, vectors_to_lists


# Define quantos resultados serão apresentados por padrão.
DEFAULT_TOP_K = 3


def load_embeddings_json(input_path: Path) -> dict[str, Any]:
    """Carrega o arquivo que contém chunks, metadados e embeddings."""

    # Informa claramente quando a etapa de embeddings ainda não foi executada.
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de embeddings não encontrado: {input_path}")

    # Abre o JSON preservando corretamente os caracteres em português.
    with input_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    # Confere se existe uma lista de registros que possa ser pesquisada.
    records = payload.get("embeddings") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("O arquivo não contém embeddings para pesquisa.")

    # Devolve o conteúdo validado para a busca.
    return payload


def encode_query(question: str, model: Any, expected_dimension: int) -> list[float]:
    """Transforma a pergunta em um vetor normalizado compatível com os documentos."""

    # Remove espaços acidentais e impede pesquisas vazias.
    cleaned_question = " ".join(question.split())
    if not cleaned_question:
        raise ValueError("A pergunta não pode ficar vazia.")

    # Gera um vetor com as mesmas opções usadas nos chunks das políticas.
    raw_vector = model.encode(
        [cleaned_question],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Converte e valida a estrutura, os números e a normalização.
    vectors = vectors_to_lists(raw_vector)
    dimension = validate_vectors(vectors, expected_count=1)

    # Impede a comparação entre vetores produzidos por modelos incompatíveis.
    if dimension != expected_dimension:
        raise ValueError(
            f"A pergunta gerou {dimension} dimensões, mas os documentos possuem "
            f"{expected_dimension}."
        )

    # Devolve o único vetor criado para a pergunta.
    return vectors[0]


def dot_product(first: list[float], second: list[float]) -> float:
    """Calcula a similaridade entre dois vetores previamente normalizados."""

    # Vetores com dimensões diferentes não podem ser comparados corretamente.
    if len(first) != len(second):
        raise ValueError("Não é possível comparar vetores com dimensões diferentes.")

    # Soma a multiplicação de cada par de posições dos dois vetores.
    return sum(left * right for left, right in zip(first, second, strict=True))


def search_embeddings(
    embeddings_payload: dict[str, Any],
    question: str,
    model: Any,
    top_k: int = DEFAULT_TOP_K,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """Ordena os chunks pela proximidade semântica com a pergunta."""

    # Confere os limites básicos da pesquisa antes de usar o modelo.
    if top_k < 1:
        raise ValueError("top_k deve ser igual ou superior a 1.")
    if min_score is not None and not -1.0 <= min_score <= 1.0:
        raise ValueError("min_score deve estar entre -1 e 1.")

    # Obtém a configuração e os registros produzidos na etapa de embeddings.
    config = embeddings_payload.get("embedding_config", {})
    records = embeddings_payload.get("embeddings", [])
    expected_dimension = int(config.get("dimension", 0))
    if expected_dimension < 1 or not records:
        raise ValueError("Configuração ou registros de embeddings inválidos.")

    # Gera o vetor da pergunta usando o mesmo modelo e dimensão dos documentos.
    question_vector = encode_query(question, model, expected_dimension)

    # Prepara uma lista que receberá cada chunk junto com sua pontuação.
    scored_results: list[dict[str, Any]] = []

    # Calcula a similaridade da pergunta com todos os chunks armazenados.
    for record in records:
        document_vector = [float(value) for value in record.get("embedding", [])]
        score = dot_product(question_vector, document_vector)

        # Descarta pontuações inválidas antes de ordenar os resultados.
        if not math.isfinite(score):
            raise ValueError(f"Pontuação inválida no chunk {record.get('chunk_id')}.")

        # Aplica o limite opcional usado futuramente para recusar respostas fracas.
        if min_score is not None and score < min_score:
            continue

        # Preserva somente os dados necessários para explicar e citar o resultado.
        scored_results.append(
            {
                "chunk_id": record["chunk_id"],
                "document_id": record["document_id"],
                "document_title": record["document_title"],
                "file_name": record["file_name"],
                "category": record["category"],
                "page": record["page"],
                "score": score,
                "text": record["text"],
            }
        )

    # Ordena da maior para a menor similaridade e mantém apenas os melhores.
    best_results = sorted(scored_results, key=lambda item: item["score"], reverse=True)[:top_k]

    # Acrescenta uma posição humana começando em primeiro lugar.
    for rank, result in enumerate(best_results, start=1):
        result["rank"] = rank

    # Entrega os chunks mais relacionados à pergunta.
    return best_results


def load_evaluation_questions(config_path: Path) -> list[dict[str, Any]]:
    """Carrega as perguntas e os documentos esperados para avaliação."""

    # Interrompe com uma mensagem direta se o arquivo não existir.
    if not config_path.exists():
        raise FileNotFoundError(f"Perguntas de avaliação não encontradas: {config_path}")

    # Lê a lista de casos de teste criada para o projeto.
    with config_path.open("r", encoding="utf-8") as stream:
        questions = json.load(stream)

    # Confere se existe ao menos uma pergunta configurada.
    if not isinstance(questions, list) or not questions:
        raise ValueError("A avaliação deve conter pelo menos uma pergunta.")

    # Garante que cada caso possua pergunta e documento esperado.
    for item in questions:
        if not item.get("question") or not item.get("expected_document_id"):
            raise ValueError("Cada avaliação precisa de pergunta e expected_document_id.")

    # Devolve os casos prontos para execução.
    return questions


def evaluate_retrieval(
    embeddings_payload: dict[str, Any],
    questions: list[dict[str, Any]],
    model: Any,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Calcula Hit@K e posição da política correta para perguntas conhecidas."""

    # Prepara os resultados detalhados e o contador de acertos.
    evaluation_results: list[dict[str, Any]] = []
    hit_count = 0

    # Executa a mesma busca semântica para cada pergunta de avaliação.
    for item in questions:
        results = search_embeddings(
            embeddings_payload,
            question=item["question"],
            model=model,
            top_k=top_k,
        )

        # Localiza a primeira posição em que apareceu o documento esperado.
        expected_document_id = item["expected_document_id"]
        matching_result = next(
            (result for result in results if result["document_id"] == expected_document_id),
            None,
        )
        hit = matching_result is not None
        if hit:
            hit_count += 1

        # Registra dados suficientes para revisar acertos e erros manualmente.
        evaluation_results.append(
            {
                "id": item.get("id"),
                "question": item["question"],
                "expected_document_id": expected_document_id,
                "hit": hit,
                "expected_document_rank": matching_result["rank"] if matching_result else None,
                "top_result_document_id": results[0]["document_id"] if results else None,
                "top_result_score": results[0]["score"] if results else None,
                "returned_chunk_ids": [result["chunk_id"] for result in results],
            }
        )

    # Calcula a proporção de perguntas com a política correta entre os resultados.
    hit_at_k = hit_count / len(questions)

    # Monta um relatório persistente e fácil de comparar após futuras mudanças.
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric": f"Hit@{top_k}",
        "summary": {
            "question_count": len(questions),
            "hit_count": hit_count,
            "miss_count": len(questions) - hit_count,
            "hit_at_k": hit_at_k,
        },
        "results": evaluation_results,
    }


def save_evaluation_json(payload: dict[str, Any], output_path: Path) -> None:
    """Salva o relatório da avaliação semântica em formato JSON."""

    # Cria a pasta de destino automaticamente quando necessário.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Grava os resultados com acentos preservados e estrutura legível.
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_retrieval_model(embeddings_payload: dict[str, Any]) -> Any:
    """Carrega exatamente o modelo usado para gerar os embeddings armazenados."""

    # Obtém o nome do modelo diretamente do arquivo para evitar incompatibilidade.
    model_name = embeddings_payload.get("embedding_config", {}).get("model_name")
    if not model_name:
        raise ValueError("O arquivo de embeddings não informa o modelo utilizado.")

    # Reutiliza o carregador central da etapa de embeddings.
    return load_sentence_transformer(model_name)
