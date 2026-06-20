import os
import pandas as pd

class UserHistoryLoader:
    def __init__(self, csv_path: str = "dataset/user_history.csv"):
        # Try both dataset/ and claims/ directory for robustness
        if not os.path.exists(csv_path):
            alt_path = "claims/user_history.csv"
            if os.path.exists(alt_path):
                csv_path = alt_path
        
        self.history_df = None
        if os.path.exists(csv_path):
            try:
                self.history_df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Error loading user history CSV {csv_path}: {e}")
        else:
            print(f"User history file not found at {csv_path}")

    def get_user_risk_context(self, user_id: str) -> str:
        if self.history_df is None:
            return "User History: No history dataset loaded."
        
        user_rows = self.history_df[self.history_df['user_id'] == user_id]
        if user_rows.empty:
            return f"User History for {user_id}:\n- Status: New user. No historical claims found."
        
        row = user_rows.iloc[0]
        # Clean and format risk values
        past_claims = row.get('past_claim_count', 0)
        accepted = row.get('accept_claim', 0)
        manual = row.get('manual_review_claim', 0)
        rejected = row.get('rejected_claim', 0)
        last_90_days = row.get('last_90_days_claim_count', 0)
        flags = row.get('history_flags', 'none')
        summary_text = row.get('history_summary', 'No summary available.')
        
        summary = (
            f"User History for {user_id}:\n"
            f"- Total Past Claims: {past_claims}\n"
            f"- Accepted: {accepted}\n"
            f"- Manual Review: {manual}\n"
            f"- Rejected: {rejected}\n"
            f"- Claims in last 90 days: {last_90_days}\n"
            f"- Historical Risk Flags: {flags}\n"
            f"- Historical Summary: {summary_text}"
        )
        return summary
