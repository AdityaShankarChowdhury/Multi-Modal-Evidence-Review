import os
import json
import hashlib

class DiskCache:
    def __init__(self, cache_dir: str = "code/.cache", cache_file: str = "gemini_cache.json"):
        # Make path relative to workspace or absolute for safety
        # Since we run from workspace root, code/.cache is appropriate
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, cache_file)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache = {}
        self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"Error loading cache from {self.cache_path}: {e}")
                self.cache = {}

    def save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving cache to {self.cache_path}: {e}")

    def resolve_image_path(self, path: str) -> str:
        # Prepend directory names to find the image
        # Semicolon can separate images, we resolve individual image paths here
        path = path.strip()
        prefixes = ["", "dataset", "claims"]
        for prefix in prefixes:
            p = os.path.join(prefix, path) if prefix else path
            if os.path.exists(p):
                return os.path.abspath(p)
        return ""

    def get_image_hash(self, path: str) -> str:
        resolved = self.resolve_image_path(path)
        if not resolved:
            # Fallback: hash the path string if file is missing to keep it deterministic
            return hashlib.sha256(path.encode('utf-8')).hexdigest()
        try:
            with open(resolved, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            print(f"Error hashing image {resolved}: {e}")
            return hashlib.sha256(path.encode('utf-8')).hexdigest()

    def generate_key(self, user_claim: str, claim_object: str, evidence_reqs: str, image_paths: list[str], strategy: str = "detailed") -> str:
        # Clean image paths list
        img_paths_clean = []
        for p in image_paths:
            if p.strip():
                img_paths_clean.append(p.strip())
        
        # Get hashes of each image, sort them to make order-independent
        img_hashes = sorted([self.get_image_hash(p) for p in img_paths_clean])
        
        # Combine parts into a structured dict
        combined = {
            "user_claim": user_claim,
            "claim_object": claim_object,
            "evidence_requirements": evidence_reqs,
            "image_hashes": img_hashes,
            "strategy": strategy
        }
        
        # Serialize deterministically and hash
        serialized = json.dumps(combined, sort_keys=True)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def get(self, key: str) -> dict | None:
        return self.cache.get(key, None)

    def set(self, key: str, value: dict):
        self.cache[key] = value
        self.save_cache()
