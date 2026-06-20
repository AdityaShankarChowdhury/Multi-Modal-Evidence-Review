import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Ensure the parent folder code/ is in Python path for local module loading
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.driver import EvidenceReviewPipeline

def main():
    print("=== Damage Claim Multi-Modal Evidence Review Pipeline ===")
    
    # Load environment variables
    load_dotenv()
    
    # 1. Load inputs
    claims_path = "dataset/claims.csv"
    if not os.path.exists(claims_path):
        claims_path = "claims/claims.csv"
        
    if not os.path.exists(claims_path):
        print(f"Error: Claims input file not found at {claims_path}!")
        return
        
    print(f"Reading input claims: {claims_path}")
    claims_df = pd.read_csv(claims_path)
    # Add a stable row-level index for resume tracking (handles duplicate user_ids)
    claims_df = claims_df.reset_index(drop=True)
    claims_df["_row_idx"] = claims_df.index
    num_rows = len(claims_df)
    print(f"Total claims to process: {num_rows}")
    
    # 2. Determine strategy
    strategy = "detailed"
    report_path = "code/evaluation/evaluation_report.md"
    if os.path.exists(report_path):
        print("Evaluation report found. Checking selected strategy...")
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
                if "MINIMAL chosen" in content or "Strategy Selection: MINIMAL" in content:
                    strategy = "minimal"
                    print("Using MINIMAL strategy as selected by evaluation.")
                else:
                    print("Using DETAILED strategy as selected by evaluation.")
        except Exception as e:
            print(f"Error reading evaluation report: {e}. Defaulting to DETAILED.")
    else:
        print("No evaluation report found. Defaulting to DETAILED strategy.")
        
    # 3. Initialize pipeline
    pipeline = EvidenceReviewPipeline(model_name="gemini-2.5-flash", rpm_limit=15, use_cache=True)
    
    # 4. Define required columns
    required_columns = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
        "issue_type", "object_part", "claim_status", "claim_status_justification",
        "supporting_image_ids", "valid_image", "severity"
    ]
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output.csv")
    progress_path = os.path.join(output_dir, "progress.csv")  # tracks by row index
    
    # Load progress
    processed_indices = set()
    if os.path.exists(progress_path):
        try:
            prog_df = pd.read_csv(progress_path)
            if "_row_idx" in prog_df.columns:
                processed_indices = set(prog_df["_row_idx"].tolist())
                print(f"Resuming: {len(processed_indices)} rows already done.")
        except Exception as e:
            print(f"Error reading progress file: {e}")
    
    # 5. Process claims incrementally by row index
    for _, row in claims_df.iterrows():
        row_idx = int(row["_row_idx"])
        user_id = str(row["user_id"])
        
        if row_idx in processed_indices:
            print(f"Skipping row {row_idx+1}/{num_rows} (User: {user_id}) - already done.")
            continue
            
        img_preview = str(row["image_paths"])[:50]
        print(f"Processing row {row_idx+1}/{num_rows} (User: {user_id}, images: {img_preview}...)...")
        res = pipeline.process_claim(
            user_id=user_id,
            image_paths_str=row["image_paths"],
            user_claim=row["user_claim"],
            claim_object=row["claim_object"],
            strategy=strategy
        )
        
        # Save with row index for deduplication-safe resume
        res_df = pd.DataFrame([res])
        res_df["_row_idx"] = row_idx
        for col in required_columns:
            if col not in res_df.columns:
                res_df[col] = "unknown"
        
        prog_exists = os.path.exists(progress_path)
        res_df.to_csv(progress_path, mode='a', header=not prog_exists, index=False)
        processed_indices.add(row_idx)
        
    # 6. Reconstruct final output in original claim order
    print("\nRebuilding final output in original claim order...")
    prog_df = pd.read_csv(progress_path)
    prog_df = prog_df.drop_duplicates(subset="_row_idx", keep="last")
    
    # Merge by row index to restore original ordering
    merged = claims_df.merge(prog_df, on="_row_idx", how="left", suffixes=("_claim", ""))
    
    # Prefer the claim's original input columns for pass-through fields
    for col in ["user_id", "image_paths", "user_claim", "claim_object"]:
        claim_col = col + "_claim"
        if claim_col in merged.columns:
            merged[col] = merged[claim_col]
    
    # Fill any missing required columns with safe defaults
    for col in required_columns:
        if col not in merged.columns:
            merged[col] = "unknown"
        merged[col] = merged[col].fillna("unknown")
    
    final_df = merged[required_columns].reset_index(drop=True)
    final_df.to_csv(output_path, index=False)
    print(f"\nSaved all {len(final_df)} predictions to: {os.path.abspath(output_path)}")
    
    # 7. Final validations
    print("\n=== Performing Final Output Validations ===")
    out_rows = len(final_df)
    print(f"Input claims row count:  {num_rows}")
    print(f"Output predictions rows: {out_rows}")
    if out_rows == num_rows:
        print("SUCCESS: Row count matches perfectly!")
    else:
        print(f"WARNING: Row count mismatch! Expected {num_rows}, got {out_rows}")
        
    if list(final_df.columns) == required_columns:
        print("SUCCESS: Column structure and order conform exactly to required schema.")
    else:
        print("WARNING: Column mismatch!")
        
    null_rows = final_df["claim_status"].isna().sum()
    if null_rows == 0:
        print("SUCCESS: All rows have claim_status populated.")
    else:
        print(f"WARNING: {null_rows} rows have missing claim_status values.")
        
    print("\nProcessing completed successfully.")

if __name__ == "__main__":
    main()
