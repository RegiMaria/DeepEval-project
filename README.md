# DeepEval-Project

Projeto de avaliação de sistemas RAG usando **DeepEval + Confident AI**.

Usa os mesmos dados do [RAGAS-Project](https://github.com/RegiMaria/RAGAS-project)
para comparar resultados entre os dois frameworks.

## O que é diferente do RAGAS-Project

| | RAGAS-Project | DeepEval-Project |
|---|---|---|
| Avaliador | RAGAS 0.1.21 | DeepEval 3.9.x |
| Juiz LLM | Claude + OpenAI obrigatória | 100% Anthropic |
| Resultado | scores 0.0–1.0 em CSV | PASSED/FAILED + reasoning |
| Visualização | CSV manual | Confident AI (dashboard web) |
| Métricas customizadas | Não | G-Eval em linguagem natural |
| CI/CD | Não | GitHub Actions + Pytest |

## Demos em vídeo

Nestes vídeos apresento na prática a diferença entre os dois frameworks — com foco no
reasoning do DeepEval, nas métricas G-Eval customizadas e no dashboard do Confident AI.

Comparamos as mesmas 5 métricas (Faithfulness, Correctness, Relevancy, Context Precision
e Context Recall) nos dois frameworks usando os mesmos dados: `historia_ia.txt` e o
golden dataset de 10 perguntas. Enquanto o RAGAS entrega scores numéricos sem explicação,
o DeepEval entrega um reasoning completo para cada falha — identificando, por exemplo,
que o retriever estava rankeando chunks irrelevantes antes do chunk correto. Além disso,
exploramos as métricas G-Eval, que permitem definir critérios em linguagem natural
específicos para o seu domínio, e vimos como o Confident AI transforma os resultados em
um dashboard navegável com histórico de runs, rastreabilidade por test case e comparação
de regressões.

- [Parte 1 — RAGAS vs DeepEval: reasoning e métricas](https://youtu.be/BId3YL5fP3I)
- [Parte 2 — Confident AI: dashboard de métricas](https://www.youtube.com/watch?v=l0YHa5kU25M)

## Estrutura

```
DeepEval-Project/
├── data/
│   ├── golden_dataset.csv         # dataset de perguntas e ground truths
│   ├── chroma_db/                 # VectorStore persistido localmente
│   └── docs/
│       └── historia_ia.txt        # documento-fonte do pipeline RAG
│
├── outputs/
│   └── golden_dataset.csv         # dataset lido pelo conftest nas avaliações
│
├── src/
│   └── deepeval_project/
│       ├── __init__.py
│       └── rag_pipeline.py        # pipeline RAG (ChromaDB + Claude Haiku)
│
├── tests/
│   ├── conftest.py                # ClaudeJudge + fixtures compartilhadas
│   ├── test_evaluate.py           # métricas equivalentes ao RAGAS
│   └── test_geval.py              # métricas customizadas G-Eval
│
├── pyproject.toml
├── requirements.txt
├── .env.example
└── .github/
    └── workflows/
        └── eval.yml               # CI/CD com GitHub Actions
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt

cp .env.example .env
# editar .env com suas chaves
```

## Dados reutilizados do RAGAS-Project

Copie os arquivos antes de rodar:

```bash
cp ../RAGAS-project/outputs/golden_dataset.csv outputs/
cp ../RAGAS-project/data/sample_docs/historia_ia.txt data/docs/
```

## Como rodar

### Localmente com pytest (só terminal, não envia ao Confident AI)

```bash
pytest tests/test_evaluate.py -v
pytest tests/test_geval.py -v
```

Roda os testes e mostra os resultados no terminal — scores, PASSED/FAILED e reasoning —
mas **não sincroniza nada com o Confident AI**.

### Com DeepEval (terminal + dashboard Confident AI)

**Passo 1 — login (uma vez só)**

```bash
deepeval login
```

Vai pedir a API key que você copia em `app.confident-ai.com → Settings → Project API Key`.

**Passo 2 — rodar com o comando do DeepEval, não com pytest direto**

```bash
deepeval test run tests/test_evaluate.py
deepeval test run tests/test_geval.py
# ou os dois de uma vez:
deepeval test run tests/
```

Esse comando faz as duas coisas: roda os testes **e** sincroniza os resultados com o
Confident AI automaticamente.

**Passo 3 — abrir o dashboard**

```bash
deepeval view
```

Abre o browser direto no dashboard com os resultados daquela execução — cada pergunta,
cada score, cada reasoning, tudo que você viu no terminal mas agora navegável e visual.

## CI/CD com GitHub Actions

O workflow em `.github/workflows/eval.yml` roda automaticamente a cada push em `main`
ou `develop`. Requer dois secrets configurados no repositório:

- `ANTHROPIC_API_KEY`
- `CONFIDENT_API_KEY`

Os resultados sobem automaticamente para `app.confident-ai.com` após cada execução.

## Confident AI

Crie uma conta gratuita em https://app.confident-ai.com

## Referências

- DeepEval: https://deepeval.com
- Confident AI: https://app.confident-ai.com
- RAGAS-Project original: https://github.com/RegiMaria/RAGAS-project
- HUYEN, Chip. *AI Engineering*. O'Reilly, 2024.
