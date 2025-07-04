import uuid
import secrets
from typing import Optional, Tuple, Union
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from .detect_pi_heuristics import detect_prompt_injection_using_heuristic_on_input
from .detect_pi_gemini import call_gemini_to_detect_pi, render_prompt_for_pi_detection
from .detect_pi_vectorbase import (
    detect_pi_using_vector_database,
    init_chroma_vector_store,
)
from secret import get_gemini_api_key
import logging

# Suppress ChromaDB telemetry warnings by patching the capture function 
try:
    import chromadb.telemetry
    chromadb.telemetry.capture = lambda *args, **kwargs: None
except Exception:
    pass

class PromptDefenseResponse(BaseModel):
    heuristic_score: float
    llm_score: float
    vector_score: float
    run_heuristic_check: bool
    run_vector_check: bool
    run_language_model_check: bool
    max_heuristic_score: float
    max_model_score: float
    max_vector_score: float
    injection_detected: bool


class PromptDefense:
    def __init__(
        self,
        gemini_api_key: str,
        chroma_collection_name: str = "prompt_defense",
    ) -> None:
        self.gemini_api_key = gemini_api_key
        self.chroma_collection_name = chroma_collection_name
        try:
            self.vector_store = init_chroma_vector_store(
                collection_name=self.chroma_collection_name
            )
        except Exception as e:
            logging.error(f"Failed to initialize Chroma vector store: {e}")
            self.vector_store = None

    def detect_injection(
        self,
        user_input: str,
        max_heuristic_score: float = 0.75,
        max_vector_score: float = 0.90,
        max_model_score: float = 0.90,
        check_heuristic: bool = True,
        check_vector: bool = True,
        check_llm: bool = True,
    ) -> PromptDefenseResponse:
        """
        Perform multi-layer prompt injection detection on user input.
        """
        try:
            heuristic_score = (
                detect_prompt_injection_using_heuristic_on_input(user_input)
                if check_heuristic
                else 0
            )
        except Exception as e:
            logging.error(f"Heuristic detection failed: {e}")
            heuristic_score = 0

        vector_score = 0
        if check_vector and self.vector_store:
            try:
                vector_score = detect_pi_using_vector_database(
                    user_input=user_input,
                    similarity_threshold=max_vector_score,
                    collection=self.vector_store,
                )["top_score"]
            except Exception as e:
                logging.error(f"Vector database detection failed: {e}")
                vector_score = 0

        model_score = 0
        if check_llm:
            try:
                prompt = render_prompt_for_pi_detection(user_input)
                model_score = call_gemini_to_detect_pi(prompt, self.gemini_api_key)
            except Exception as e:
                logging.error(f"LLM detection failed: {e}")
                model_score = 0

        injection_detected = (
            heuristic_score > max_heuristic_score
            or vector_score > max_vector_score
            or model_score > max_model_score
        )

        return PromptDefenseResponse(
            heuristic_score=heuristic_score,
            llm_score=model_score,
            vector_score=vector_score,
            run_heuristic_check=check_heuristic,
            run_vector_check=check_vector,
            run_language_model_check=check_llm,
            max_heuristic_score=max_heuristic_score,
            max_model_score=max_model_score,
            max_vector_score=max_vector_score,
            injection_detected=injection_detected,
        )

    @staticmethod
    def generate_canary_word(length: int = 8) -> str:
        """Generate a random hex canary token."""
        return secrets.token_hex(length // 2)

    def add_canary_word(
        self,
        prompt: Union[str, PromptTemplate],
        canary_word: Optional[str] = None,
        canary_format: str = "<!-- {canary_word} -->",
    ) -> Tuple[Union[str, PromptTemplate], str]:
        """
        Adds a canary token to the prompt to track leakage.
        """
        canary_word = canary_word or self.generate_canary_word()
        canary_comment = canary_format.format(canary_word=canary_word)

        if isinstance(prompt, str):
            return canary_comment + "\n" + prompt, canary_word
        elif isinstance(prompt, PromptTemplate):
            prompt.template = canary_comment + "\n" + prompt.template
            return prompt, canary_word
        else:
            raise TypeError(f"Invalid prompt type: {type(prompt)}")

    def is_canary_word_leaked(
        self,
        user_input: str,
        completion: str,
        canary_word: str,
    ) -> bool:
        """Check if the LLM leaked the canary token in its output."""
        return canary_word in completion

    def log_leakage(
        self,
        user_input: str,
        completion: str,
        canary_word: str,
    ) -> None:
        """
        Log the input, completion, and canary word in ChromaDB for future vector matching.
        """
        self.vector_store.add(
            documents=[user_input],
            metadatas=[
                {
                    "completion": completion,
                    "canary_word": canary_word,
                }
            ],
        )

    def detect_prompt_injection_and_log(prompt: str) -> str:
        defense = PromptDefense(gemini_api_key=..., chroma_collection_name="prompt_defense")
        result = defense.detect_injection(prompt)
        if result.injection_detected:
            defense.vector_store.add(documents=[prompt])
            return "injection detected"
        return "no injection"


def evaluate_prompt_safety(prompt: str) -> dict:
    """
    Evaluates the safety of a prompt using ChromaDB, heuristic, and LLM layers.
    Returns a JSON dict: {'result': 'success'/'fail', 'data': prompt or failure message}
    """
    try:
        defense = PromptDefense(gemini_api_key=get_gemini_api_key())
        # 1. Check for similar prompt in ChromaDB (vectorbase), but only if there is data
        if defense.vector_store:
            try:
                vector_result = defense.vector_store.query(query_texts=[prompt], n_results=1)
                distances = vector_result.get("distances", [[]])[0]
                if distances:  # Only check if there is data in ChromaDB
                    top_score = 1 - distances[0]
                    if top_score > 0.8:
                        return {"result": "fail", "data": "Prompt injection detected (vectorbase match)."}
            except Exception as e:
                logging.error(f"Vectorbase query failed: {e}")
        # 2. Run multi-layer detection
        result = defense.detect_injection(prompt)
        if result.llm_score > 0.7:
            if defense.vector_store:
                try:
                    defense.vector_store.add(documents=[prompt], ids=[str(uuid.uuid4())])
                except Exception as e:
                    logging.error(f"Failed to add to vector store: {e}")
            return {"result": "fail", "data": "Prompt injection detected."}
        if result.llm_score > 0.5 and result.heuristic_score > result.max_heuristic_score:
            if defense.vector_store:
                try:
                    defense.vector_store.add(documents=[prompt], ids=[str(uuid.uuid4())])
                except Exception as e:
                    logging.error(f"Failed to add to vector store: {e}")
            return {"result": "fail", "data": "Prompt injection detected."}
        return {"result": "success", "data": prompt}
    except Exception as e:
        logging.error(f"Prompt safety evaluation failed: {e}")
        return {"result": "error", "data": str(e)}
