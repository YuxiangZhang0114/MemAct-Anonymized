import json
import random
import openai
from typing import Dict, Any


def extract_solution_str(solution_str: str) -> str:
    marker = "FINAL_ANSWER: "
    idx = solution_str.rfind(marker)
    if idx == -1:
        return ""
    result = solution_str[idx + len(marker):]
    result = result.replace("<|im_end|>", "").strip()
    return result

def create_evaluation_prompt(question: str, ground_truth: str, model_answer: str) -> str:
    """Create a prompt for evaluation."""
    prompt = f"""You are a highly skilled and precise answer evaluation assistant. Your task is to meticulously compare a model's response to a standard answer for a given question and judge their semantic equivalence.

The question may contain multiple sub-questions. Evaluate each sub-question independently before making a final judgment.

## Given Question(s):
{question}

## Standard Answer(s):
<start>
{ground_truth}
<end>

## Model Response:
<start>
{model_answer}
<end>

## Evaluation Criteria:
(C1) **Semantic Equivalence**: The core meaning must be the same. Different phrasing is acceptable as long as the essential information is conveyed correctly.
(C2) **Formatting**: Ignore differences in formatting (e.g., markdown), capitalization, and punctuation.
(C3) **Numerical Precision**: For numerical answers, allow for reasonable precision differences (e.g., "3.14" and "3.14159" should be considered equivalent for most contexts).
(C4) **Multiple Choice**: For multiple-choice questions, the selected option(s) must be identical to the standard answer.
(C5) **Extraneous Content**: If the model's response contains the correct answer but also includes additional harmless content (like "Certainly, the answer is..."), it is still considered correct. However, if the additional content contradicts the correct answer or introduces factual errors, it must be judged as incorrect.
(C6) **Empty Response**: An empty or non-responsive answer (e.g., "I don't know") is incorrect.

## Final Judgment Rules:
- If all parts of the model's response are semantically equivalent to the standard answer, the final judgment is **CORRECT**.
- If the question contains multiple sub-questions, and the model correctly answers at least one sub-question but fails on at least one other, the final judgment is **PARTIAL_CORRECT**.
- In all other cases, including when the entire response is wrong or contains contradictions (per C5), the final judgment is **INCORRECT**.

Please provide your final judgment by outputting ONLY one of the following tags. Do not add any explanation or other text.

<ANSWER_CORRECT>
<ANSWER_PARTIAL_CORRECT>
<ANSWER_INCORRECT>

Judgment Result: """
    
    return prompt

def compute_score(solution_str: str, extra_info: dict) -> float:
    """Evaluate answer correctness using a model.
    
    Returns 1.0 if correct, otherwise 0.0.
    """

    model_answer = extract_solution_str(solution_str)
    if len(model_answer) == 0:
        return 0.0
    
    question = extra_info["question"]
    ground_truth = extra_info["ground_truth"]

    # Configure OpenAI client to connect to SGLang service
    client = openai.OpenAI(
        base_url="http://x.x.x.x:9527/v1",  # SGLang service address
        api_key="dummy-key"  # SGLang doesn't require a real API key, but one must be provided
    )

    # Create evaluation prompt
    evaluation_prompt = create_evaluation_prompt(question, ground_truth, model_answer)

    score = 0.0
    model_response = None
    try:
        response = client.chat.completions.create(
            model="gpt-oss",  # Use the model name configured in the service
            messages=[
                {"role": "user", "content": evaluation_prompt}
            ],
            max_tokens=800,
            temperature=1.0,  # Use deterministic output
            stream=False
        )
        
        model_response = response.choices[0].message.content.strip()
        
        # Determine model evaluation result
        if "<ANSWER_CORRECT>" in model_response.split("final<|message|>")[-1]:
            score = 1.0
        else:
            score = 0.0
            
    except Exception as e:
        # Network or service failure → default to 0.0
        model_response = f"Evaluation failed: {str(e)}"
        score = 0.0

    # Optional: log evaluation for debugging/analysis
    try:
        with open("mem_agent_model_eval.json", "a", encoding="utf-8") as f:
            json.dump(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer,
                    "evaluation_prompt": evaluation_prompt,
                    "model_response": model_response,
                    "score": score,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
    except Exception:
        pass

    return float(score)

if __name__ == "__main__":
    extra_info = {
        "question": "Harold tosses a nickel four times. Probability of at least as many heads as tails?",
        "ground_truth": "11/16",
    }
    # Test evaluation
    result = compute_score("FINAL_ANSWER: 11/16", extra_info)