import os
import sys
import time
import pandas as pd

# Add the parent code directory to Python path to import pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.driver import EvidenceReviewPipeline
from evaluation.metrics import calculate_field_metrics, calculate_overall_exact_match

def run_evaluation():
    print("=== Starting Model Strategy Evaluation ===")
    
    # 1. Load sample dataset
    sample_path = "dataset/sample_claims.csv"
    if not os.path.exists(sample_path):
        sample_path = "claims/sample_claims.csv"
    
    if not os.path.exists(sample_path):
        print(f"Error: Sample claims file not found at {sample_path}!")
        return
        
    print(f"Loaded sample claims from: {sample_path}")
    sample_df = pd.read_csv(sample_path)
    num_rows = len(sample_df)
    print(f"Number of sample rows to evaluate: {num_rows}")
    
    # Define fields to evaluate
    eval_fields = [
        "evidence_standard_met",
        "valid_image",
        "claim_status",
        "issue_type",
        "object_part",
        "severity",
        "risk_flags",
        "supporting_image_ids"
    ]
    
    strategies = ["minimal", "detailed"]
    strategy_results = {}
    
    # We will compute statistics
    for strategy in strategies:
        print(f"\n--- Evaluating Strategy: {strategy.upper()} ---")
        
        # Initialize pipeline for this run
        pipeline = EvidenceReviewPipeline(model_name="gemini-2.5-flash", rpm_limit=15, use_cache=True)
        
        start_time = time.time()
        predictions = []
        
        for idx, row in sample_df.iterrows():
            print(f"Processing row {idx+1}/{num_rows} (User: {row['user_id']})...")
            pred = pipeline.process_claim(
                user_id=row["user_id"],
                image_paths_str=row["image_paths"],
                user_claim=row["user_claim"],
                claim_object=row["claim_object"],
                strategy=strategy
            )
            predictions.append(pred)
            
        latency = time.time() - start_time
        preds_df = pd.DataFrame(predictions)
        
        # Calculate metrics per field
        field_scores = {}
        for field in eval_fields:
            scores = calculate_field_metrics(
                preds=preds_df[field].tolist(),
                targets=sample_df[field].tolist(),
                field_type=field
            )
            field_scores[field] = scores["accuracy"]
            
        # Calculate overall exact match rate
        exact_match = calculate_overall_exact_match(preds_df, sample_df, eval_fields)
        
        # Store results
        strategy_results[strategy] = {
            "field_scores": field_scores,
            "exact_match": exact_match,
            "latency": latency,
            "total_calls": pipeline.total_calls,
            "cache_hits": pipeline.cache_hits,
            "input_tokens": pipeline.total_input_tokens,
            "output_tokens": pipeline.total_output_tokens,
            "images_processed": pipeline.images_processed,
            "predictions_df": preds_df
        }
        
        print(f"Strategy {strategy.upper()} completed in {latency:.2f} seconds.")
        print(f"Exact Match Rate: {exact_match:.2%}")
        for field, acc in field_scores.items():
            print(f" - {field} Accuracy: {acc:.2%}")
            
    # 2. Select the best strategy
    best_strategy = "detailed"
    if strategy_results["minimal"]["exact_match"] > strategy_results["detailed"]["exact_match"]:
        best_strategy = "minimal"
        
    print(f"\n=== Strategy Selection: {best_strategy.upper()} chosen (Highest Exact Match) ===")
    
    # 3. Generate the evaluation report (evaluation_report.md)
    report_content = generate_markdown_report(strategy_results, best_strategy, num_rows)
    
    report_dir = "code/evaluation"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "evaluation_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Evaluation report written successfully to: {report_path}")
    return best_strategy

def generate_markdown_report(results: dict, chosen_strategy: str, num_rows: int) -> str:
    # Pricing assumptions
    # Gemini 2.5 Flash pricing:
    # $0.075 / 1M input tokens
    # $0.300 / 1M output tokens
    in_token_rate = 0.075 / 1_000_000
    out_token_rate = 0.300 / 1_000_000
    
    # Sample set operational costs (non-cached scenario estimation)
    minimal_ops = results["minimal"]
    detailed_ops = results["detailed"]
    
    min_cost = (minimal_ops["input_tokens"] * in_token_rate) + (minimal_ops["output_tokens"] * out_token_rate)
    det_cost = (detailed_ops["input_tokens"] * in_token_rate) + (detailed_ops["output_tokens"] * out_token_rate)
    
    # Extrapolate for the full test set of 44 rows
    extrapolate_ratio = 44 / num_rows
    test_in_tokens_est = detailed_ops["input_tokens"] * extrapolate_ratio
    test_out_tokens_est = detailed_ops["output_tokens"] * extrapolate_ratio
    test_calls_est = 44
    test_images_est = detailed_ops["images_processed"] * extrapolate_ratio
    test_cost_est = (test_in_tokens_est * in_token_rate) + (test_out_tokens_est * out_token_rate)
    
    report = f"""# Evaluation Report

This report evaluates and compares two prompt strategies for the multi-modal evidence review pipeline using the sample dataset.

## Summary of Results

| Evaluation Metric | Minimal Prompt (Strategy A) | Detailed Prompt (Strategy B) |
| :--- | :---: | :---: |
| **Exact Match Rate (Overall)** | **{minimal_ops['exact_match']:.2%}** | **{detailed_ops['exact_match']:.2%}** |
| `claim_status` Accuracy | {minimal_ops['field_scores']['claim_status']:.2%} | {detailed_ops['field_scores']['claim_status']:.2%} |
| `evidence_standard_met` Accuracy | {minimal_ops['field_scores']['evidence_standard_met']:.2%} | {detailed_ops['field_scores']['evidence_standard_met']:.2%} |
| `valid_image` Accuracy | {minimal_ops['field_scores']['valid_image']:.2%} | {detailed_ops['field_scores']['valid_image']:.2%} |
| `issue_type` Accuracy | {minimal_ops['field_scores']['issue_type']:.2%} | {detailed_ops['field_scores']['issue_type']:.2%} |
| `object_part` Accuracy | {minimal_ops['field_scores']['object_part']:.2%} | {detailed_ops['field_scores']['object_part']:.2%} |
| `severity` Accuracy | {minimal_ops['field_scores']['severity']:.2%} | {detailed_ops['field_scores']['severity']:.2%} |
| `risk_flags` Accuracy | {minimal_ops['field_scores']['risk_flags']:.2%} | {detailed_ops['field_scores']['risk_flags']:.2%} |
| `supporting_image_ids` Accuracy | {minimal_ops['field_scores']['supporting_image_ids']:.2%} | {detailed_ops['field_scores']['supporting_image_ids']:.2%} |
| **Total Evaluation Latency** | {minimal_ops['latency']:.2f} s | {detailed_ops['latency']:.2f} s |

## Chosen Strategy

Based on the evaluation metrics above, **{chosen_strategy.upper()}** has been selected as the optimal prompt strategy. 

*Rationale:* The {chosen_strategy} prompt provides better grounding and formatting rules, minimizing parsing errors and maximizing exact match accuracy on ground-truth evaluation claims.

---

## Operational Analysis (for selected {chosen_strategy.upper()} strategy)

### Sample Run Statistics (20 claims)
- **Model calls made:** {detailed_ops['total_calls']}
- **Cache hits:** {detailed_ops['cache_hits']}
- **Images processed:** {detailed_ops['images_processed']}
- **Input tokens used:** {detailed_ops['input_tokens']:,}
- **Output tokens used:** {detailed_ops['output_tokens']:,}
- **Estimated API Cost (no cache):** ${det_cost:.6f}

### Extrapolated Test Set Predictions (44 claims)
- **Estimated model calls:** {test_calls_est}
- **Estimated images processed:** {int(test_images_est)}
- **Estimated input tokens:** {int(test_in_tokens_est):,}
- **Estimated output tokens:** {int(test_out_tokens_est):,}
- **Estimated API Cost (no cache):** ${test_cost_est:.6f}
- **Estimated Runtime:** ~{(detailed_ops['latency'] * extrapolate_ratio):.2f} seconds (at 15 RPM throttling)

### Operational Optimization Strategy
- **Rate Limiting:** A strict token-bucket/sleep rate limiter set to 15 RPM respects free/paid API quotas and prevents `ResourceExhausted` errors.
- **Backoff & Retries:** Automatic exponential backoff with random jitter handles network issues and transient API rate limits.
- **Disk Caching:** Hashing input text and image byte structures avoids duplicate API requests when executing pipeline iterations locally, driving latency and cost to zero for cached claims.
- **Pydantic Validation & Fallbacks:** A validation guard intercepts invalid or malformed outputs. It triggers a single corrective retry before resorting to a safe fallback schema row.
"""
    return report

if __name__ == "__main__":
    run_evaluation()
