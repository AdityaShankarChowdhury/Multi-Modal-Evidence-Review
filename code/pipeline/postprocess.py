import json
from pydantic import ValidationError
from pipeline.schema import (
    ClaimAnalysis, RiskFlagType, IssueTypeType, ClaimStatusType, SeverityType, ObjectPartType
)

# Lists of allowed values for validation and coercion
CLAIM_STATUSES = ["supported", "contradicted", "not_enough_information"]
ISSUE_TYPES = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown"
]
CAR_PARTS = [
    "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
    "headlight", "taillight", "fender", "quarter_panel", "body", "unknown"
]
LAPTOP_PARTS = [
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base", "body", "unknown"
]
PACKAGE_PARTS = [
    "box", "package_corner", "package_side", "seal", "label", "contents", "item", "unknown"
]
RISK_FLAGS = [
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required"
]
SEVERITIES = ["none", "low", "medium", "high", "unknown"]

def get_fallback_row(user_id: str, image_paths: str, user_claim: str, claim_object: str, reason: str = "Validation failure") -> dict:
    """
    Returns a safe default dictionary matching the expected output schema.
    """
    return {
        "user_id": user_id,
        "image_paths": image_paths,
        "user_claim": user_claim,
        "claim_object": claim_object,
        "evidence_standard_met": False,
        "evidence_standard_met_reason": f"System error or validation failure: {reason}",
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": f"Unable to process claim evidence due to validation errors ({reason}). Requires manual inspection.",
        "supporting_image_ids": "none",
        "valid_image": False,
        "severity": "unknown"
    }

def coerce_value(val: str, allowed: list, default: str) -> str:
    """Helper to coerce a string value to lowercase and match against allowed values."""
    if not val:
        return default
    v = str(val).strip().lower()
    if v in allowed:
        return v
    # Try substring matches or prefix matches
    for item in allowed:
        if item in v or v in item:
            return item
    return default

def clean_risk_flags(flags: list) -> str:
    """Cleans a list of risk flags and joins them with semicolons."""
    if not flags:
        return "none"
    
    cleaned = []
    for f in flags:
        cf = coerce_value(f, RISK_FLAGS, "manual_review_required")
        if cf not in cleaned:
            cleaned.append(cf)
            
    # Handle redundant 'none' when other flags are present
    if "none" in cleaned and len(cleaned) > 1:
        cleaned.remove("none")
        
    if not cleaned:
        return "none"
        
    return ";".join(cleaned)

def clean_supporting_image_ids(image_ids: list) -> str:
    """Cleans a list of supporting image IDs and joins them with semicolons."""
    if not image_ids:
        return "none"
    
    cleaned = []
    for img_id in image_ids:
        c_id = str(img_id).strip()
        if c_id.lower() in ["", "none", "null"]:
            continue
        # Remove extension if present
        if "." in c_id:
            c_id = c_id.split(".")[0]
        if c_id not in cleaned:
            cleaned.append(c_id)
            
    if not cleaned:
        return "none"
    return ";".join(cleaned)

def clean_object_part(part: str, claim_object: str) -> str:
    """Validates and coerces object part depending on the claim object type."""
    obj = str(claim_object).lower().strip()
    if obj == "car":
        return coerce_value(part, CAR_PARTS, "unknown")
    elif obj == "laptop":
        return coerce_value(part, LAPTOP_PARTS, "unknown")
    elif obj == "package":
        return coerce_value(part, PACKAGE_PARTS, "unknown")
    else:
        return "unknown"

def process_and_validate_response(
    response_str: str,
    user_id: str,
    image_paths: str,
    user_claim: str,
    claim_object: str
) -> dict | None:
    """
    Parses, validates, and coerces the raw string JSON from Gemini.
    Returns:
        dict: The final dictionary mapping to output CSV format if validation succeeds.
        None: If validation fails completely (indicates a retry is needed).
    """
    try:
        data = json.loads(response_str)
    except Exception as e:
        print(f"Failed to parse raw JSON from Gemini response: {e}")
        return None

    # We validate using Pydantic first
    try:
        analysis = ClaimAnalysis.model_validate(data)
    except ValidationError as e:
        print(f"Pydantic Validation Error: {e}")
        # Return None to trigger corrective retry, or we can handle minor field failures
        return None
    except Exception as e:
        print(f"Validation Error: {e}")
        return None

    # Perform coercion and serialization to CSV-ready format
    try:
        final_row = {
            "user_id": user_id,
            "image_paths": image_paths,
            "user_claim": user_claim,
            "claim_object": claim_object,
            
            # Direct fields
            "evidence_standard_met": bool(analysis.evidence_standard_met),
            "evidence_standard_met_reason": str(analysis.evidence_standard_met_reason).strip(),
            
            # Cleaned/coerced fields
            "risk_flags": clean_risk_flags(analysis.risk_flags),
            "issue_type": coerce_value(analysis.issue_type, ISSUE_TYPES, "unknown"),
            "object_part": clean_object_part(analysis.object_part, claim_object),
            "claim_status": coerce_value(analysis.claim_status, CLAIM_STATUSES, "not_enough_information"),
            
            "claim_status_justification": str(analysis.claim_status_justification).strip(),
            "supporting_image_ids": clean_supporting_image_ids(analysis.supporting_image_ids),
            "valid_image": bool(analysis.valid_image),
            "severity": coerce_value(analysis.severity, SEVERITIES, "unknown")
        }
        return final_row
    except Exception as e:
        print(f"Error during postprocess formatting: {e}")
        return None
