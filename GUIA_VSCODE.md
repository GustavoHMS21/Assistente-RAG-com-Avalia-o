# Como executar o ProcedIA no VS Code

## Fluxo completo

Abra a pasta **Projeto RH** e use **Terminal > Executar Tarefa**.

1. **Preparar ambiente** — cria `.venv` e instala as bibliotecas.
2. **Extrair políticas dos PDFs** — gera `policy_pages.json`.
3. **Criar chunks pesquisáveis** — gera `policy_chunks.json`.
4. **Gerar embeddings locais** — gera `policy_embeddings.json`.
5. **Pesquisar políticas** — mostra os chunks semanticamente mais próximos.
6. **Avaliar busca semântica** — calcula Hit@3 com perguntas conhecidas.
7. **Perguntar ao assistente de RH** — recupera evidências e gera uma resposta pelo Ollama local.
8. **Executar todos os testes** — verifica todas as etapas do projeto.
9. **Abrir interface web** — abre a versão em navegador (Streamlit) em http://localhost:8501.

## Configurar o Ollama

O projeto usa o [Ollama](https://ollama.com) rodando localmente, sem necessidade de chave de API.

1. Instale o Ollama e mantenha o aplicativo aberto (ou rode `ollama serve`).
2. Baixe o modelo usado pelo projeto:

```powershell
ollama pull qwen3.5:4b
```

Opcionalmente, o `.env` pode conter um endereço diferente do padrão:

```text
OLLAMA_HOST=http://localhost:11434
```

## Fazer uma pergunta completa

Execute a tarefa **7. Perguntar ao assistente de RH** e digite, por exemplo:

```text
Quantos dias de férias posso vender?
```

O fluxo completo fará:

1. Embedding da pergunta.
2. Busca dos três chunks mais próximos.
3. Verificação do limite de relevância.
4. Envio apenas da pergunta e das evidências ao modelo local via Ollama.
5. Geração da resposta em português.
6. Apresentação das fontes calculadas localmente.

Também é possível usar o terminal:

```powershell
python ask_policies.py "Quantos dias de férias posso vender?"
```

## Usar a interface web

Execute a tarefa **9. Abrir interface web** (ou `streamlit run app.py` no terminal) e acesse http://localhost:8501 no navegador. A tela mostra um campo para a pergunta, um botão "Perguntar" e, no painel lateral, ajustes de quantos trechos buscar e o limite mínimo de relevância. A resposta e as fontes aparecem na própria página, sem precisar do terminal.

## Proteções implementadas

- Nenhuma chave de API é usada; tudo roda localmente.
- O modelo recebe somente os chunks recuperados.
- O prompt proíbe invenção de regras e interpretação jurídica.
- As fontes são montadas pelo código, não inventadas pelo modelo.
- Perguntas abaixo do limite de relevância são recusadas sem chamada ao Ollama.
- Falha de conexão com o Ollama ou modelo ausente geram mensagens seguras.

## Desempenho local

Como a geração roda na sua máquina (CPU, salvo se houver GPU compatível), as respostas podem demorar mais do que em uma API na nuvem, especialmente na primeira chamada de cada sessão, quando o modelo é carregado na memória.

## Próxima etapa

Refinar a interface Streamlit (histórico de perguntas, exportação de respostas).
