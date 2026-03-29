# DeepEval-Project

Projeto de avaliação de sistemas RAG usando **DeepEval + Confident AI**.

Usa os mesmos dados do [RAGAS-Project](https://github.com/RegiMaria/RAGAS-project)
para comparar resultados entre os dois frameworks.

## O que é diferente do RAGAS-Project

| | RAGAS-Project | DeepEval-Project |
|---|---|---|
| Avaliador | RAGAS 0.1.21 | DeepEval >= 0.21 |
| Juiz LLM | Claude + OpenAI obrigatória | 100% Anthropic |
| Resultado | scores 0.0–1.0 em CSV | PASSED/FAILED + reasoning |
| Visualização | CSV manual | Confident AI (dashboard web) |
| Métricas customizadas | Não | G-Eval em linguagem natural |
| CI/CD | Não | GitHub Actions + Pytest |

## Estrutura

```
DeepEval-Project/
├── data/
│   ├── golden_dataset.csv     # copiado do RAGAS-Project/outputs/
│   └── historia_ia.txt        # copiado do RAGAS-Project/data/sample_docs/
│
├── tests/
│   ├── test_evaluate.py       # 5 métricas equivalentes ao RAGAS
│   └── test_geval.py          # métricas customizadas G-Eval
│
├── conftest.py                # ClaudeJudge + fixtures compartilhadas
├── requirements.txt
├── .env.example
└── .github/
    └── workflows/
        └── eval.yml           # CI/CD com GitHub Actions
```

## Dados reutilizados do RAGAS-Project

Copie os arquivos antes de rodar:

```bash
cp ../RAGAS-project/outputs/golden_dataset.csv data/
cp ../RAGAS-project/data/sample_docs/historia_ia.txt data/
```

## Instalação

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt

cp .env.example .env
# editar .env com suas chaves
```

## Uso

```bash
# 1. conectar ao Confident AI (uma vez só)
deepeval login

# 2. rodar todos os testes
deepeval test run tests/

# 3. abrir o dashboard no browser
deepeval view
```

## Confident AI

Os resultados sobem automaticamente para `app.confident-ai.com` após cada execução.
Crie uma conta gratuita em https://app.confident-ai.com

## Referências

- DeepEval: https://deepeval.com
- Confident AI: https://app.confident-ai.com
- RAGAS-Project original: https://github.com/RegiMaria/RAGAS-project
- HUYEN, Chip. *AI Engineering*. O'Reilly, 2024.
