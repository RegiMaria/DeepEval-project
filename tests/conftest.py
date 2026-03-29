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
    Carrega o golden_dataset.csv gerado pelo RAGAS-Project.
    Arquivo esperado em: data/golden_dataset.csv

    Colunas (mesmo formato do RAGAS-Project):
        question      → input do LLMTestCase
        answer        → actual_output (resposta do RAG)
        contexts      → retrieval_context (chunks recuperados)
        ground_truth  → expected_output
    """
    df = pd.read_csv("outputs/golden_dataset.csv")

    # contexts vem como string no CSV — converter para lista
    df["contexts"] = df["contexts"].apply(
        lambda x: x.split("|||") if isinstance(x, str) else [str(x)]
    )
    return df
