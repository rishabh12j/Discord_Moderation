import torch
from transformers import pipeline

class ToxicityJudge:
    """
    Wraps the XLM-R large toxicity classifier to provide baseline reward signals
    for the Reinforcement Learning agent.
    """
    def __init__(self, model_name: str = "textdetox/xlmr-large-toxicity-classifier-v2"):
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Initialize the pipeline with truncation to handle long sequences safely
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            device=self.device,
            truncation=True,
            max_length=512
        )

    def score_text(self, text: str) -> float:
        """
        Processes native multilingual text and returns the toxicity probability.
        Strictly bounds the output float to the 0.0 to 1.0 range.
        """
        if not text.strip():
            return 0.0
            
        result = self.classifier(text)[0]
        label = result['label'].lower()
        score = result['score']

        # Ensure the returned metric specifically represents the probability of toxicity
        if label == 'toxic':
            toxicity_prob = score
        else:
            toxicity_prob = 1.0 - score
            
        return float(max(0.0, min(toxicity_prob, 1.0)))

if __name__ == "__main__":
    # Rapid validation test
    judge = ToxicityJudge()
    test_score = judge.score_text("This is a standard test message.")
    print(f"Test Score: {test_score:.4f}")