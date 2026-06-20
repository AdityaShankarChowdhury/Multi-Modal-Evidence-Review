import os
import json
from pipeline.history import UserHistoryLoader
from pipeline.evidence_rules import EvidenceRequirementsLoader
from pipeline.cache import DiskCache
from pipeline.gemini_client import GeminiClient
from pipeline.prompt_builder import PromptBuilder
from pipeline.postprocess import process_and_validate_response, get_fallback_row
from pipeline.schema import ClaimAnalysis

class EvidenceReviewPipeline:
    def __init__(self, model_name: str = "gemini-2.5-flash", rpm_limit: int = 15, use_cache: bool = True):
        self.history_loader = UserHistoryLoader()
        self.evidence_loader = EvidenceRequirementsLoader()
        self.cache = DiskCache()
        self.client = GeminiClient(model_name=model_name, rpm_limit=rpm_limit)
        self.prompt_builder = PromptBuilder()
        self.use_cache = use_cache
        
        # Token and operational statistics
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.cache_hits = 0
        self.images_processed = 0

    def process_claim(self, user_id: str, image_paths_str: str, user_claim: str, claim_object: str, strategy: str = "detailed") -> dict:
        """
        Orchestrates processing for a single claim row.
        """
        # Parse image paths
        image_paths = [p.strip() for p in image_paths_str.split(";") if p.strip()]
        self.images_processed += len(image_paths)
        
        # 1. Fetch risk history and evidence requirements
        history_text = self.history_loader.get_user_risk_context(user_id)
        evidence_text = self.evidence_loader.get_requirements_for_claim(claim_object)
        
        # 2. Check disk cache
        cache_key = self.cache.generate_key(user_claim, claim_object, evidence_text, image_paths, strategy=strategy)
        if self.use_cache:
            cached_res = self.cache.get(cache_key)
            if cached_res is not None:
                self.cache_hits += 1
                return cached_res
        
        # 3. Build prompt and parts
        prompt_text, image_parts = self.prompt_builder.build_prompt_and_parts(
            user_claim=user_claim,
            claim_object=claim_object,
            evidence_requirements_text=evidence_text,
            user_history_text=history_text,
            image_paths=image_paths,
            strategy=strategy
        )
        
        # Combine instructions and binary image parts
        contents = [prompt_text] + image_parts
        
        # 4. Make Gemini API Call (attempt 1)
        self.total_calls += 1
        
        try:
            raw_response = self.client.generate_structured_content(
                contents=contents,
                response_schema=ClaimAnalysis
            )
            
            # Approximate token logging (typical token size is roughly 1 token per 4 chars for English)
            # Multimodal images in Gemini 2.5 Flash count as 258 tokens per image
            self.total_input_tokens += len(prompt_text) // 4 + len(image_parts) * 258
            self.total_output_tokens += len(raw_response) // 4
            
            final_row = process_and_validate_response(
                response_str=raw_response,
                user_id=user_id,
                image_paths=image_paths_str,
                user_claim=user_claim,
                claim_object=claim_object
            )
            
            if final_row is not None:
                if self.use_cache:
                    self.cache.set(cache_key, final_row)
                return final_row
                
        except Exception as e:
            print(f"Error on first API attempt for {user_id}: {e}")
            raw_response = None
            
        # 5. Corrective follow-up retry on schema failure
        print(f"Corrective retry triggered for user {user_id}...")
        try:
            corrective_instruction = (
                "\n\nCRITICAL: Your previous response failed strict Pydantic schema validation. "
                "Ensure that risk_flags contains only items from the allowed set, object_part matches the "
                "appropriate object type list, and the output is a valid JSON object matching the requested schema."
            )
            contents = [prompt_text + corrective_instruction] + image_parts
            
            raw_response = self.client.generate_structured_content(
                contents=contents,
                response_schema=ClaimAnalysis
            )
            
            self.total_input_tokens += (len(prompt_text) + len(corrective_instruction)) // 4 + len(image_parts) * 258
            self.total_output_tokens += len(raw_response) // 4
            
            final_row = process_and_validate_response(
                response_str=raw_response,
                user_id=user_id,
                image_paths=image_paths_str,
                user_claim=user_claim,
                claim_object=claim_object
            )
            
            if final_row is not None:
                if self.use_cache:
                    self.cache.set(cache_key, final_row)
                return final_row
                
        except Exception as e:
            print(f"Error on corrective follow-up API retry for {user_id}: {e}")
            
        # 6. Fallback safe row if both attempts fail
        print(f"Schema validation failed twice for {user_id}. Using default fallback row.")
        fallback = get_fallback_row(
            user_id=user_id,
            image_paths=image_paths_str,
            user_claim=user_claim,
            claim_object=claim_object,
            reason="Gemini schema validation failed twice."
        )
        return fallback
