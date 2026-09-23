# ProcedIA

Protótipo RAG de um auxiliar de Recursos Humanos que consulta políticas internas e produz respostas acompanhadas das fontes utilizadas.

## Arquitetura atual

```text
PDFs
  ↓
Extração e validação
  ↓
Chunking com metadados
  ↓
Embeddings locais
  ↓
Busca semântica Top-3
  ↓
Proteção por relevância
  ↓
Resposta fundamentada pelo Ollama (local)
  ↓
Fontes verificadas localmente
```

## Tecnologias

- Python
- pypdf
- Sentence Transformers
- `paraphrase-multilingual-MiniLM-L12-v2`
- Ollama (servidor local de LLMs)
- Modelo `qwen3.5:4b`
- Streamlit
- unittest

## Segurança e fundamentação

- O modelo roda 100% local via Ollama; nenhuma chave de API é necessária.
- O endereço do servidor Ollama é configurável via `.env` (opcional, `OLLAMA_HOST`).
- A geração recebe somente os chunks recuperados.
- Uma pontuação mínima evita chamadas quando a evidência é fraca.
- A aplicação cria a lista de fontes a partir dos metadados originais.
- O modelo é instruído a não inventar regras nem interpretar juridicamente.

## Estrutura principal

```text
Projeto RH/
├── Politicas/
├── config/
├── data/processed/
├── src/
│   ├── answerer.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── pdf_reader.py
│   ├── retriever.py
│   └── settings.py
├── tests/
├── app.py
├── ask_policies.py
├── build_chunks.py
├── build_embeddings.py
├── evaluate_search.py
├── extract_policies.py
├── search_policies.py
└── requirements.txt
```

## Executar

No VS Code, use **Terminal > Executar Tarefa** e siga as tarefas numeradas de 1 a 9.

Para uma pergunta completa pelo terminal:

```powershell
python ask_policies.py "Qual é o prazo para enviar um atestado?"
```

Para abrir a interface web:

```powershell
streamlit run app.py
```

## Resultado atual

- 3 documentos e 9 páginas.
- 27 chunks e 27 embeddings.
- 384 dimensões por embedding.
- Hit@3 de 100% nas cinco perguntas de recuperação.
- 17 testes aprovados.
- Geração fundamentada implementada e protegida por limite de relevância.

## Requisitos locais

- [Ollama](https://ollama.com) instalado e em execução.
- Modelo baixado com `ollama pull qwen3.5:4b`.

## Próxima etapa

Refinar a interface Streamlit (histórico de perguntas, exportação de respostas).
