"""
conftest.py
Carregado automaticamente pelo Pytest antes de qualquer teste.

ClaudeJudge: juiz LLM 100% Anthropic.
Sem OpenAI, sem LangchainLLMWrapper, sem ValueError.
"""

import pytest
import pandas as pd
import anthropic
from deepeval.models import DeepEvalBaseLLM


# ─────────────────────────────────────────────────────────
# Juiz LLM
# ─────────────────────────────────────────────────────────

class ClaudeJudge(DeepEvalBaseLLM):
    """
    Juiz LLM usando Claude Sonnet via Anthropic.

    No RAGAS-Project, o llm_factory do RAGAS 0.3.x/0.4.x
    só aceitava OpenAI nativamente — qualquer tentativa com Claude
    gerava ValueError: Collections metrics only support modern InstructorLLM.

    Aqui não há esse problema: o DeepEval aceita qualquer LLM
    que implemente esta interface.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model_name = model
        self.client = anthropic.Anthropic()

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


# ─────────────────────────────────────────────────────────
# Fixtures compartilhadas
# ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def judge():
    """
    Instancia o ClaudeJudge uma única vez por sessão.
    scope="session" evita criar N conexões com a API Anthropic.
    """
    return ClaudeJudge()


@pytest.fixture(scope="session")
def golden_dataset():
    """
    Carrega o golden_dataset.csv e enriquece com answer e contexts
    gerados pelo pipeline RAG em tempo de execução.

    Colunas do CSV:
        question       → input do LLMTestCase
        ground_truth   → expected_output
        question_type  → informativo (simple, multi_context…)

    Colunas adicionadas pelo pipeline:
        answer         → actual_output (resposta gerada pelo RAG)
        contexts       → retrieval_context (chunks recuperados)
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from deepeval_project.rag_pipeline import RAGPipeline

    df = pd.read_csv("outputs/golden_dataset.csv")

    rag = RAGPipeline()
    rag.build()

    answers = []
    contexts_list = []
    for question in df["question"]:
        result = rag.query(question)
        answers.append(result["answer"])
        contexts_list.append(result["contexts"])

    df["answer"] = answers
    df["contexts"] = contexts_list
    return df
