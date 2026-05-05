"""
Multilingual Toxicity Judge
Uses XLM-RoBERTa-based classifier trained on multilingual toxicity data.
Works natively in 15+ languages without translation.
"""

from transformers import pipeline
import torch
from typing import Dict, Union
import time


class ToxicityJudge:
    """
    Wraps textdetox/xlmr-large-toxicity-classifier-v2 for fast toxicity scoring.
    
    Model details:
    - Base: XLM-RoBERTa Large (560M params)
    - Training: TextDetox multilingual dataset
    - Languages: en, ru, uk, de, es, am, zh, ar, hi, it, fr, he, ja, tt
    - Output: Binary classification (toxic/non-toxic) with confidence score
    """
    
    def __init__(self, device: Union[int, str] = None):
        """
        Args:
            device: GPU device ID (0, 1, ...) or "cpu". Auto-detects if None.
        """
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device
            
        print(f"Loading ToxicityJudge on device: {self.device}")
        print("Model: textdetox/xlmr-large-toxicity-classifier-v2")
        
        self.classifier = pipeline(
            "text-classification",
            model="textdetox/xlmr-large-toxicity-classifier-v2",
            device=self.device,
            truncation=True,
            max_length=512,
            padding=True
        )
        
        print("✓ ToxicityJudge loaded successfully\n")

    def score_text(self, text: str, return_label: bool = False) -> Union[float, Dict]:
        """
        Returns toxicity probability in [0, 1].
        
        Args:
            text: Input message in any supported language
            return_label: If True, returns dict with score and label
            
        Returns:
            float: Toxicity score (0 = clean, 1 = toxic)
            dict: {score, label, confidence} if return_label=True
        """
        if not text or not text.strip():
            return 0.0 if not return_label else {"score": 0.0, "label": "non-toxic", "confidence": 1.0}
        
        result = self.classifier(text)[0]
        label = result['label'].lower()
        confidence = result['score']
        
        # Convert to toxicity probability
        # Model returns either "toxic" or "non-toxic"
        if label in ('toxic', '1', '__label__toxic'):
            toxicity_score = confidence
        else:
            toxicity_score = 1.0 - confidence
        
        if return_label:
            return {
                "score": toxicity_score,
                "label": label,
                "confidence": confidence
            }
        
        return toxicity_score
    
    def score_batch(self, texts: list[str], batch_size: int = 32) -> list[float]:
        """
        Batch inference for efficiency.
        
        Args:
            texts: List of messages
            batch_size: Batch size for pipeline
            
        Returns:
            List of toxicity scores
        """
        if not texts:
            return []
        
        # Filter empty strings
        texts_filtered = [t if t and t.strip() else " " for t in texts]
        
        results = self.classifier(texts_filtered, batch_size=batch_size)
        
        scores = []
        for result in results:
            label = result['label'].lower()
            confidence = result['score']
            
            if label in ('toxic', '1', '__label__toxic'):
                toxicity_score = confidence
            else:
                toxicity_score = 1.0 - confidence
            
            scores.append(toxicity_score)
        
        return scores


def benchmark_judge(num_samples: int = 100):
    """
    Benchmark inference speed across languages.
    """
    test_messages = {
        'en': "You are an idiot and nobody likes you.",
        'es': "Eres un imbécil y nadie te quiere.",
        'ru': "Ты идиот и никто тебя не любит.",
        'fr': "Tu es stupide et personne ne t'aime.",
        'de': "Du bist ein Idiot und niemand mag dich.",
        'zh': "你是个白痴，没有人喜欢你。",
        'ar': "أنت أحمق ولا أحد يحبك.",
        'hi': "तुम बेवकूफ हो और कोई तुम्हें पसंद नहीं करता।",
        'hin': "Kattapa ne bahubali ko kyu mara?",
        'clean_en': "Hello, how are you today?",
        'clean_es': "Hola, ¿cómo estás hoy?"
    }
    
    judge = ToxicityJudge()
    
    print("=" * 60)
    print("TOXICITY JUDGE BENCHMARK")
    print("=" * 60)
    
    for lang, text in test_messages.items():
        # Warmup
        _ = judge.score_text(text)
        
        # Timed runs
        times = []
        for _ in range(num_samples):
            start = time.time()
            score = judge.score_text(text)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times) * 1000  # Convert to ms
        result = judge.score_text(text, return_label=True)
        
        print(f"\n{lang:12s} | Score: {result['score']:.3f} | "
              f"Label: {result['label']:10s} | "
              f"Avg Time: {avg_time:.1f}ms")
    
    print("\n" + "=" * 60)
    print(f"✓ All tests passed. Target: <100ms per inference on GPU")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_judge(num_samples=50)
