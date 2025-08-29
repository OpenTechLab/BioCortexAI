# chunk_corpus.py
import sentencepiece as spm
import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

def main():
    # ... (načtení tokenizéru a tokenizace celého souboru do all_tokens) ...
    print("Načítám SentencePiece model...")
    sp = spm.SentencePieceProcessor()
    sp.load(f'{config.model_prefix}.model')
    
    all_tokens = []
    print(f"Načítám a tokenizuji soubor '{config.preprocessed_text_path}'...")
    with open(config.preprocessed_text_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if line:
                token_ids = [sp.bos_id()] + sp.encode_as_ids(line) + [sp.eos_id()]
                all_tokens.extend(token_ids)

    # Rozdělení tokenů na tréninkovou a validační sadu (např. 95% / 5%)
    n = len(all_tokens)
    train_tokens = all_tokens[:int(n*0.95)]
    val_tokens = all_tokens[int(n*0.95):]
    
    print(f"Rozděluji na {len(train_tokens)} tréninkových a {len(val_tokens)} validačních tokenů.")

    # Zpracování a uložení tréninkových dat
    num_chunks_train = len(train_tokens) // config.pretrain_max_seq_len
    train_data = np.array(train_tokens[:num_chunks_train * config.pretrain_max_seq_len], dtype=np.uint16)
    train_data = train_data.reshape(num_chunks_train, config.pretrain_max_seq_len)
    
    train_filename = config.chunked_binary_path.replace('.bin', '_train.bin')
    train_data.tofile(train_filename)
    print(f"Tréninková data uložena do: {train_filename}")

    # Zpracování a uložení validačních dat
    num_chunks_val = len(val_tokens) // config.pretrain_max_seq_len
    val_data = np.array(val_tokens[:num_chunks_val * config.pretrain_max_seq_len], dtype=np.uint16)
    val_data = val_data.reshape(num_chunks_val, config.pretrain_max_seq_len)
    
    val_filename = config.chunked_binary_path.replace('.bin', '_val.bin')
    val_data.tofile(val_filename)
    print(f"Validační data uložena do: {val_filename}")

    print("Hotovo.")

if __name__ == "__main__":
    main()