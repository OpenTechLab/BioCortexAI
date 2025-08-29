# prepare_data.py
import os
import sentencepiece as spm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

def train_sentencepiece_model():
    """
    Trénuje model SentencePiece na základě vstupních dat a uloží jej.
    """
    # Zajištění, že vstupní soubor existuje
    if not os.path.exists(config.preprocessed_text_path):
        print(f"Chyba: Vstupní soubor '{config.preprocessed_text_path}' ...")
        return

    print(f"Trénuji SentencePiece model na souboru: {config.preprocessed_text_path}")
    
    # OPRAVA: Použití cesty z config.py
    spm.SentencePieceTrainer.train(
        f'--input={config.preprocessed_text_path} --model_prefix={config.model_prefix} '
        f'--vocab_size={config.vocab_size} --model_type={config.model_type} '
        '--bos_id=0 --eos_id=1 --unk_id=2 --pad_id=-1' # Pad token by měl mít ID, které se nepoužije jinde. -1 je pro ignore_index ideální.
    )

if __name__ == '__main__':
    train_sentencepiece_model()
    print("SentencePiece model byl úspěšně natrénován a uložen.")