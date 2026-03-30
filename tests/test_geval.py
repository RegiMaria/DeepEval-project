"""
tests/test_geval.py
Métricas G-Eval customizadas para o domínio do historia_ia.txt.

Estas métricas não existem no RAGAS-Project.
Capturam erros que o RAGAS marcava como corretos porque
usava similaridade matemática, não julgamento de domínio.

Problemas documentados no RAGAS-Project que estas métricas resolvem:

  1. Variância entre execuções (0.81 vs 0.86):
     → completude_resposta tem critério explícito, resultado estável

  2. Faithfulness 1.00 mesmo com conhecimento externo:
     → sem_conhecimento_externo detecta quando o LLM vai além do documento

  3. Erros factuais históricos com score alto:
     → precisao_historica penaliza datas e nomes errados explicitamente
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval


# ─────────────────────────────────────────────────────────
# Métricas G-Eval
# ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def precisao_historica(judge):
    """
    Detecta respostas com datas, nomes ou eventos históricos incorretos.

    Problema no RAGAS-Project: o RAGAS calculava similaridade semântica —
    "anos 50" e "1950" são semanticamente próximos, score alto.
    Mas "Conferência de Dartmouth em 1954" (errado) vs "1956" (correto)
    também são próximos e o RAGAS não pegava o erro.
    """
    return GEval(
        name="precisao_historica",
        criteria="""
        Avalie se as informações históricas na resposta estão corretas.
        Verifique: datas, nomes de pesquisadores, nomes de marcos tecnológicos
        e sequência cronológica dos eventos sobre história da Inteligência Artificial.
        Qualquer erro factual — mesmo que a resposta pareça semanticamente correta —
        deve resultar em score baixo.
        Score 1.0 = todos os fatos históricos corretos e verificáveis.
        Score 0.0 = contém pelo menos um erro factual histórico.
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.85,
        model=judge,
    )


@pytest.fixture(scope="module")
def sem_conhecimento_externo(judge):
    """
    Detecta quando o RAG usa conhecimento do LLM além do documento.

    Problema no RAGAS-Project: Faithfulness marcava 1.00 em casos onde
    o LLM completava respostas com conhecimento geral sobre IA que não
    estava no historia_ia.txt. O RAGAS só verificava se as afirmações
    podiam ser inferidas do contexto — não se vinham de outro lugar.
    """
    return GEval(
        name="sem_conhecimento_externo",
        criteria="""
        Verifique se a resposta contém APENAS informações presentes
        no contexto recuperado do documento.
        O sistema RAG deve responder somente com base no texto fornecido —
        não com conhecimento geral sobre Inteligência Artificial.
        Se a resposta inclui detalhes corretos mas ausentes no contexto
        recuperado, isso é uma falha: o LLM está usando conhecimento externo.
        Score 1.0 = resposta 100% baseada no contexto recuperado.
        Score 0.0 = resposta usa informações fora do contexto.
        """,
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.9,
        model=judge,
    )


@pytest.fixture(scope="module")
def completude_resposta(judge):
    """
    Detecta respostas parciais que o RAGAS marcava como corretas.

    Problema no RAGAS-Project: respostas que cobriam apenas parte
    do ground truth geravam scores oscilando entre execuções (0.80, 0.86)
    porque a similaridade semântica é sensível a pequenas variações.
    Aqui o critério é explícito: omissões são penalizadas, extras não.
    """
    return GEval(
        name="completude_resposta",
        criteria="""
        Verifique se a resposta cobre TODOS os aspectos relevantes da pergunta
        com base no ground truth fornecido.
        Penalize respostas que omitem partes importantes do que foi perguntado.
        Não penalize por informações adicionais corretas.
        Score 1.0 = cobre completamente o que o ground truth especifica.
        Score 0.0 = omite aspectos centrais da resposta esperada.
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.75,
        model=judge,
    )


# ─────────────────────────────────────────────────────────
# Testes
# ─────────────────────────────────────────────────────────

def test_precisao_historica(precisao_historica, golden_dataset):
    """
    Garante que o RAG não erra fatos históricos mesmo quando
    a resposta parece semanticamente correta ao RAGAS.
    """
    for _, row in golden_dataset.iterrows():
        tc = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            retrieval_context=row["contexts"],
            expected_output=row["ground_truth"],
        )
        assert_test(tc, [precisao_historica])


def test_sem_conhecimento_externo(sem_conhecimento_externo, golden_dataset):
    """
    Detecta alucinação fundamentada — o LLM completa com
    conhecimento próprio mesmo quando o contexto é suficiente.
    Este é o caso que o RAGAS-Project marcava como Faithfulness 1.00.
    """
    for _, row in golden_dataset.iterrows():
        tc = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            retrieval_context=row["contexts"],
            expected_output=row["ground_truth"],
        )
        assert_test(tc, [sem_conhecimento_externo])


def test_completude_resposta(completude_resposta, golden_dataset):
    """
    Resolve o problema de variância do RAGAS-Project.
    Respostas parciais tinham scores oscilando entre execuções.
    Aqui o critério é explícito — omissões são sempre penalizadas.
    """
    for _, row in golden_dataset.iterrows():
        tc = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            retrieval_context=row["contexts"],
            expected_output=row["ground_truth"],
        )
        assert_test(tc, [completude_resposta])
