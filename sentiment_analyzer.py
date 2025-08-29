# sentiment_analyzer.py
import torch
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):

        try:
            print("Sentiment analyzer initializating...")
            device_index = 0 if torch.cuda.is_available() else -1
            if device_index == 0:
                print("Using CUDA: cuda:0") #try to avoid CPU
            else:
                print("Using CPU") #watch CPU temperature

            self.classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                device=device_index  # 0 => CUDA; -1 => CPU
            )
            print("Sentiment analyzer is ready.")
        except Exception as e:
            print(f"ERROR: Can not load the sentiment model: {e}")
            print("Try run: pip install torch transformers sentencepiece protobuf")
            self.classifier = None

    def get_score(self, text: str) -> float:

        if not self.classifier or not text:
            return 0.0

        try:
            truncated_text = text
            text_bez_user = truncated_text.replace("user: ", "")
            clean_text = text_bez_user.replace("model: ", "")
            print(f"Text do analyzeru sentimentu: {clean_text}");
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