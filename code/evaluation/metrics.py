def clean_bool(val) -> bool:
    """Standardizes boolean values from string or boolean formats."""
    if isinstance(val, bool):
        return val
    val_str = str(val).strip().lower()
    return val_str in ["true", "1", "yes", "t"]

def clean_set(val) -> set:
    """Splits a semicolon-separated string into a set of normalized strings."""
    if not val or val is None:
        return {"none"}
    val_str = str(val).strip().lower()
    parts = [p.strip() for p in val_str.split(";") if p.strip()]
    if not parts or parts == ["none"] or parts == [""]:
        return {"none"}
    return set(parts)

def clean_str(val) -> str:
    """Normalizes string inputs to lowercase with stripped whitespace."""
    if val is None:
        return "unknown"
    return str(val).strip().lower()

def calculate_field_metrics(preds: list, targets: list, field_type: str) -> dict:
    """
    Computes accuracy for a specific field based on its comparison semantics.
    """
    if not preds or not targets or len(preds) != len(targets):
        return {"accuracy": 0.0}

    correct_count = 0
    total = len(preds)

    for p, t in zip(preds, targets):
        is_correct = False
        if field_type in ["evidence_standard_met", "valid_image"]:
            is_correct = clean_bool(p) == clean_bool(t)
        elif field_type in ["risk_flags", "supporting_image_ids"]:
            is_correct = clean_set(p) == clean_set(t)
        else:
            is_correct = clean_str(p) == clean_str(t)
        
        if is_correct:
            correct_count += 1

    accuracy = correct_count / total
    return {"accuracy": accuracy}

def calculate_overall_exact_match(preds_df, targets_df, evaluated_fields: list) -> float:
    """
    Calculates exact-match rate across all evaluated fields per row.
    """
    correct_rows = 0
    total = len(preds_df)

    for idx in range(total):
        row_correct = True
        for field in evaluated_fields:
            p = preds_df.loc[idx, field]
            t = targets_df.loc[idx, field]
            
            if field in ["evidence_standard_met", "valid_image"]:
                is_correct = clean_bool(p) == clean_bool(t)
            elif field in ["risk_flags", "supporting_image_ids"]:
                is_correct = clean_set(p) == clean_set(t)
            else:
                is_correct = clean_str(p) == clean_str(t)
                
            if not is_correct:
                row_correct = False
                break
                
        if row_correct:
            correct_rows += 1

    return correct_rows / total if total > 0 else 0.0
