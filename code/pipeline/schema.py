from pydantic import BaseModel, Field
from typing import List, Literal

# Allowed values defined as Type Literals for strict parsing & client-side type safety
RiskFlagType = Literal[
    'none', 'blurry_image', 'cropped_or_obstructed', 'low_light_or_glare',
    'wrong_angle', 'wrong_object', 'wrong_object_part', 'damage_not_visible',
    'claim_mismatch', 'possible_manipulation', 'non_original_image',
    'text_instruction_present', 'user_history_risk', 'manual_review_required'
]

IssueTypeType = Literal[
    'dent', 'scratch', 'crack', 'glass_shatter', 'broken_part', 'missing_part',
    'torn_packaging', 'crushed_packaging', 'water_damage', 'stain', 'none', 'unknown'
]

ClaimStatusType = Literal['supported', 'contradicted', 'not_enough_information']

SeverityType = Literal['none', 'low', 'medium', 'high', 'unknown']

ObjectPartType = Literal[
    # Car parts
    'front_bumper', 'rear_bumper', 'door', 'hood', 'windshield', 'side_mirror',
    'headlight', 'taillight', 'fender', 'quarter_panel', 'body',
    # Laptop parts
    'screen', 'keyboard', 'trackpad', 'hinge', 'lid', 'corner', 'port', 'base',
    # Package parts
    'box', 'package_corner', 'package_side', 'seal', 'label', 'contents', 'item',
    # Default/Unknown
    'unknown'
]

class ClaimAnalysis(BaseModel):
    evidence_standard_met: bool = Field(
        description="true if the image set satisfies the minimum evidence requirements to evaluate the claim; otherwise false."
    )
    evidence_standard_met_reason: str = Field(
        description="A short explanation of why the evidence standard was met or not, grounded in the images and rules."
    )
    risk_flags: List[RiskFlagType] = Field(
        description="List of risk flags from the allowed set. Return ['none'] if no flags apply. If multiple risks apply, list all."
    )
    issue_type: IssueTypeType = Field(
        description="The visible issue type. Use 'none' if object is undamaged. Use 'unknown' if it cannot be determined."
    )
    object_part: ObjectPartType = Field(
        description="The relevant part of the object. Must belong to the list of parts matching the claim_object type (car, laptop, package)."
    )
    claim_status: ClaimStatusType = Field(
        description="supported: visual evidence supports user claim; contradicted: visual evidence directly contradicts; not_enough_information: evidence is missing/insufficient."
    )
    claim_status_justification: str = Field(
        description="Concise, image-grounded explanation for the status decision, referencing image IDs when appropriate."
    )
    supporting_image_ids: List[str] = Field(
        description="List of image IDs (e.g. ['img_1']) that contain visual evidence supporting this decision. Return ['none'] if no images are supporting."
    )
    valid_image: bool = Field(
        description="true if the image set is usable for automated review (unblurred, correct object, enough light, not manipulated); otherwise false."
    )
    severity: SeverityType = Field(
        description="none (undamaged), low, medium, high, or unknown (if not possible to tell)."
    )
