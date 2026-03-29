"""
tests/test_evaluate.py
As 5 métricas equivalentes às do RAGAS-Project.

Comparação direta:

    RAGAS-Project (evaluate.py)      DeepEval-Project (este arquivo)
    ─────────────────────────────    ──────────────────────────────────────
    Faithfulness                 →   FaithfulnessMetric
    Answer Correctness           →   AnswerCorrectnessMetric
    Answer Relevance             →   AnswerRelevancyMetric
    Context Precision            →   ContextualPrecisionMetric
    Context Recall               →   ContextualRecallMetric

Diferenças chave:
  - Resultado é PASSED/FAILED, não um score solto
  - Cada falha vem com reasoning explicando o motivo
  - Não depende de OpenAI para embeddings (Answer Relevancy)
  - Resultados sobem automaticamente ao Confident AI
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerCorrectnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)


# ─────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────

def build_test_cases(golden_dataset):
    """Converte o golden_dataset em lista de LLMTestCase."""
    test_cases = []
    for _, row in golden_dataset.iterrows():
        test_cases.append(LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            retrieval_context=row["contexts"],
            expected_output=row["ground_truth"],
        ))
    return test_cases


# ─────────────────────────────────────────────────────────
# Testes — um por métrica para isolar falhas
# ─────────────────────────────────────────────────────────

def test_faithfulness(judge, golden_dataset):
    """
    RAGAS equivalente: Faithfulness
    Pergunta: a resposta está baseada no contexto ou o LLM inventou?

    No RAGAS-Project, execução 1 deu 1.00 e execução 2 deu 0.81.
    Aqui o threshold 0.7 resolve a variância: ambas seriam PASSED.
    """
    metric = FaithfulnessMetric(threshold=0.7, model=judge)
    for tc in build_test_cases(golden_dataset):
        assert_test(tc, [metric])


def test_correctness(judge, golden_dataset):
    """
    RAGAS equivalente: Answer Correctness
    Pergunta: a resposta corresponde ao ground truth?

    No RAGAS-Project: execução 1 = 0.86, execução 2 = 0.80.
    """
    metric = AnswerCorrectnessMetric(threshold=0.7, model=judge)
    for tc in build_test_cases(golden_dataset):
        assert_test(tc, [metric])


def test_relevancy(judge, golden_dataset):
    """
    RAGAS equivalente: Answer Relevance
    Pergunta: a resposta é relevante para a pergunta feita?

    Diferença crítica: o RAGAS 0.1.21 usava OpenAI Embeddings
    internamente (text-embedding-ada-002) para calcular esta métrica,
    independente da configuração. O DeepEval usa o juiz LLM — sem OpenAI.
    """
    metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
    for tc in build_test_cases(golden_dataset):
        assert_test(tc, [metric])


def test_context_precision(judge, golden_dataset):
    """
    RAGAS equivalente: Context Precision
    Pergunta: os chunks recuperados são precisos para a pergunta?
    """
    metric = ContextualPrecisionMetric(threshold=0.7, model=judge)
    for tc in build_test_cases(golden_dataset):
        assert_test(tc, [metric])


def test_context_recall(judge, golden_dataset):
    """
    RAGAS equivalente: Context Recall
    Pergunta: o contexto cobriu o necessário para a resposta correta?
    """
    metric = ContextualRecallMetric(threshold=0.7, model=judge)
    for tc in build_test_cases(golden_dataset):
        assert_test(tc, [metric])


def test_suite_completa(judge, golden_dataset):
    """
    Equivalente ao evaluate() do RAGAS — todas as métricas de uma vez.
    Use para ter o panorama geral antes de investigar falhas individuais.
    Os resultados desta run ficam no Confident AI para comparar com runs futuras.
    """
    metrics = [
        FaithfulnessMetric(threshold=0.7, model=judge),
        AnswerCorrectnessMetric(threshold=0.7, model=judge),
        AnswerRelevancyMetric(threshold=0.7, model=judge),
        ContextualPrecisionMetric(threshold=0.7, model=judge),
        ContextualRecallMetric(threshold=0.7, model=judge),
    ]
    for tc in build_test_cases(golden_dataset):
        assert_test(tc, metrics)
