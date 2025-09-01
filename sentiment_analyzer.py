# sentiment_analyzer.py
import torch
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        print("Sentiment analyzer preparing (lazy init)...")
        self.classifier = None
        self.device_index = 0 if torch.cuda.is_available() else -1
        if self.device_index == 0:
            print("Will use CUDA when needed")
        else:
            print("Will use CPU when needed")

    def lazy_init(self) -> bool:
        """Initialize the model only when needed"""
        if self.classifier is None:
            try:
                print("Loading sentiment model...")
                self.classifier = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                    device=self.device_index
                )
                print("Sentiment model loaded.")
                return True
            except Exception as e:
                print(f"ERROR: Cannot load sentiment model: {e}")
                print("Try run: pip install torch transformers sentencepiece protobuf")
                return False
        return True

    def get_score(self, text: str) -> float:
        if not text:
            return 0.0

        # Lazy initialization
        if not self.lazy_init():
            return 0.0

        try:
            text_bez_user = text.replace("user: ", "")
            clean_text = text_bez_user.replace("model: ", "")
            result = self.classifier(clean_text)[0]

            score = result.get('score', 0.0)
            label = result.get('label', 'neutral')

            if label.lower() == 'negative':
                return -score
            elif label.lower() == 'positive':
                return score
            else: # neutral
                return 0.0
        except Exception:
            print("Sentiment EXCEPTION!!!")
            return 0.0

sentiment_analyzer = SentimentAnalyzer()

if __name__ == '__main__':
    #loner demo
    text1 = "Tohle je naprosto úžasný den!"
    text2 = "Vůbec se mi to nelíbí, je to hrozné."
    text3 = "Dnes je úterý."
    
    print("\n--- Sentiment analyzer test ---")
    print(f"'{text1}' -> Skóre: {sentiment_analyzer.get_score(text1):.3f}")
    print(f"'{text2}' -> Skóre: {sentiment_analyzer.get_score(text2):.3f}")
    print(f"'{text3}' -> Skóre: {sentiment_analyzer.get_score(text3):.3f}")
