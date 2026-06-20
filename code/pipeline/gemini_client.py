import os
import time
import random
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load env variables from code/.env or standard system env
load_dotenv()

# We can specify the config file if loaded inside code/
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

class GeminiClient:
    def __init__(self, model_name: str = "gemini-2.5-flash", rpm_limit: int = 15):
        self.model_name = model_name
        self.rpm_limit = rpm_limit
        self.request_interval = 60.0 / rpm_limit if rpm_limit > 0 else 0
        self.last_request_time = 0.0
        
        # Read API key from GEMINI_API_KEY env var
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("WARNING: GEMINI_API_KEY environment variable is not set. API calls will fail unless configured globally.")
            
        # Initialize official google-genai client
        # In the new SDK, genai.Client() automatically picks up GEMINI_API_KEY, 
        # but passing it explicitly makes it robust and transparent.
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()

    def generate_structured_content(self, contents, response_schema, max_retries: int = 5) -> str:
        # Rate limit throttling (enforces strict RPM limit)
        if self.request_interval > 0:
            elapsed = time.time() - self.last_request_time
            sleep_time = self.request_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.last_request_time = time.time()

        base_delay = 2.0
        for attempt in range(max_retries):
            try:
                # Call Gemini API using standard model and schema options
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.0,  # Strict temperature=0.0 for maximum determinism
                    )
                )
                
                # Check response text
                if not response.text:
                    raise ValueError("Gemini API returned an empty text response.")
                
                return response.text

            except Exception as e:
                err_msg = str(e)
                print(f"Gemini API attempt {attempt+1} failed: {err_msg}")
                
                # If we are at the last attempt, raise the exception
                if attempt == max_retries - 1:
                    print("Max retries reached. Raising API error.")
                    raise e
                
                # Apply exponential backoff with jitter
                sleep_duration = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                print(f"Retrying in {sleep_duration:.2f} seconds...")
                time.sleep(sleep_duration)
        
        raise RuntimeError("Unexpected end of retry loop without return or error.")
