import pytest
import re
from src.rag_engine import FinancialRAGEngine

SAMPLE_TEST_SET = [
    {
        "question": "What was Apple's total net sales in 2023?",
        "expected_value": "383,285",  # In millions
        "ticker": "AAPL"
    }
]

def test_table_exact_match_metric():
    engine = FinancialRAGEngine()
    correct = 0
    
    for case in SAMPLE_TEST_SET:
        answer, citations = engine.query(case["question"])
        # Check if the exact expected numeric string appears in the response
        clean_expected = case["expected_value"].replace(",", "")
        clean_answer = answer.replace(",", "")
        
        if clean_expected in clean_answer or case["expected_value"] in answer:
            correct += 1

    accuracy = (correct / len(SAMPLE_TEST_SET)) * 100
    print(f"\nEvaluated {len(SAMPLE_TEST_SET)} questions. Exact-match accuracy: {accuracy:.2f}%")
    assert accuracy >= 70.0