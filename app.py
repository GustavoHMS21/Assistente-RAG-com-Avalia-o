"""Interface web do ProcedIA: reúne pergunta, resposta e fontes em uma tela simples."""

# Permite usar recursos modernos de tipagem sem avaliá-los imediatamente.
from __future__ import annotations

# Importa ferramentas para caminhos de arquivos.
from pathlib import Path

# Importa o Streamlit, responsável por transformar este script em uma página web.
import streamlit as st

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
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "policy_embeddings.json"


# Define o título da aba do navegador e o ícone exibido.
st.set_page_config(page_title="ProcedIA", page_icon="📋")


@st.cache_resource(show_spinner="Carregando base de políticas...")
def get_embeddings_payload() -> dict:
    """Carrega o arquivo de embeddings uma única vez por sessão do servidor."""

    # Reaproveita a mesma função usada pelo script de terminal.
    return load_embeddings_json(EMBEDDINGS_PATH)


@st.cache_resource(show_spinner="Carregando modelo de busca semântica...")
def get_retrieval_model(_embeddings_payload: dict):
    """Carrega o modelo de embeddings uma única vez por sessão do servidor."""

    # O prefixo "_" no parâmetro evita que o Streamlit tente gerar um hash do dicionário grande.
    return load_retrieval_model(_embeddings_payload)


@st.cache_resource(show_spinner=False)
def get_ollama_client():
    """Cria o cliente do Ollama uma única vez por sessão do servidor."""

    return create_ollama_client()


# Título e descrição apresentados no topo da página.
st.title("📋 ProcedIA")
st.caption("Assistente de consulta às políticas internas de RH, com respostas fundamentadas em fontes verificáveis.")

# Painel lateral com os ajustes finos, mantendo a tela principal simples.
with st.sidebar:
    st.header("Configurações")
    top_k = st.slider("Quantidade de trechos buscados", min_value=1, max_value=5, value=DEFAULT_TOP_K)
    min_score = st.slider(
        "Limite mínimo de relevância",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_MIN_RELEVANCE_SCORE,
        step=0.05,
        help="Perguntas cuja evidência mais próxima fique abaixo deste valor são recusadas sem chamar o modelo.",
    )
    st.caption(f"Modelo gerador: `{DEFAULT_OLLAMA_MODEL}` (via Ollama local)")

# Campo onde a pessoa digita a pergunta.
question = st.text_input("Digite sua pergunta sobre as políticas de RH:")

# O botão evita gerar uma resposta a cada tecla digitada.
if st.button("Perguntar", type="primary") and question.strip():
    try:
        # Carrega (ou reaproveita do cache) os componentes pesados do pipeline.
        embeddings_payload = get_embeddings_payload()
        retrieval_model = get_retrieval_model(embeddings_payload)
        client = get_ollama_client()

        # Mostra um indicador de carregamento enquanto o modelo local gera a resposta.
        with st.spinner("Consultando as políticas e gerando a resposta..."):
            results = search_embeddings(
                embeddings_payload,
                question=question,
                model=retrieval_model,
                top_k=top_k,
            )
            answer_payload = generate_grounded_answer(
                question,
                results=results,
                client=client,
                model=DEFAULT_OLLAMA_MODEL,
                min_relevance_score=min_score,
            )
    except Exception as exc:
        # Mostra um erro amigável sem expor detalhes técnicos desnecessários.
        st.error(friendly_api_error(exc))
    else:
        # Apresenta a resposta final em destaque.
        st.subheader("Resposta")
        if answer_payload["refused"]:
            st.warning(answer_payload["answer"])
        else:
            st.write(answer_payload["answer"])

        # Apresenta as fontes recuperadas, calculadas localmente pelo código.
        if answer_payload["sources"]:
            st.subheader("Fontes recuperadas")
            for number, source in enumerate(answer_payload["sources"], start=1):
                st.markdown(
                    f"**Fonte {number}:** {source['document_title']} | "
                    f"página {source['page']} | similaridade `{source['score']:.4f}` | "
                    f"`{source['chunk_id']}`"
                )

        st.caption(f"Modelo gerador: {answer_payload['model']}")
