import os
import pandas as pd

class EvidenceRequirementsLoader:
    def __init__(self, csv_path: str = "dataset/evidence_requirements.csv"):
        # Try both dataset/ and claims/ directory for robustness
        if not os.path.exists(csv_path):
            alt_path = "claims/evidence_requirements.csv"
            if os.path.exists(alt_path):
                csv_path = alt_path
        
        self.rules_df = None
        if os.path.exists(csv_path):
            try:
                self.rules_df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Error loading evidence requirements CSV {csv_path}: {e}")
        else:
            print(f"Evidence requirements file not found at {csv_path}")

    def get_requirements_for_claim(self, claim_object: str) -> str:
        if self.rules_df is None:
            return "Minimum Evidence Requirements: No rules dataset loaded."
        
        # Filter for rules matching the claim_object or 'all'
        # Check matching case insensitively
        claim_object_lower = str(claim_object).lower()
        filtered_df = self.rules_df[
            (self.rules_df['claim_object'].str.lower() == claim_object_lower) |
            (self.rules_df['claim_object'].str.lower() == 'all')
        ]
        
        if filtered_df.empty:
            return f"Minimum Evidence Requirements: None configured for '{claim_object}'."
        
        lines = [f"Minimum Evidence Requirements for {claim_object.upper()}:"]
        for _, row in filtered_df.iterrows():
            req_id = row.get('requirement_id', 'REQ')
            applies = row.get('applies_to', 'general')
            details = row.get('minimum_image_evidence', '')
            lines.append(f"- [{req_id}] Applies to '{applies}': {details}")
        
        return "\n".join(lines)
