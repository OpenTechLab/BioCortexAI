# preprocess_corpus.py
import os
import glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config 

# Vstupní a výstupní cesty
# Můžete je dát do configu nebo nechat zde
INPUT_DIR = config.raw_data_dir
OUTPUT_FILE = config.preprocessed_text_path

def main():
    # Ujistěte se, že vstupní adresář existuje
    if not os.path.isdir(INPUT_DIR):
        print(f"Chyba: Adresář '{INPUT_DIR}' neexistuje. Vytvořte ho a vložte do něj syrové .txt soubory.")
        # Vytvoření ukázkového souboru pro demonstraci
        os.makedirs(INPUT_DIR, exist_ok=True)
        with open(os.path.join(INPUT_DIR, "ukazka.txt"), "w", encoding="utf-8") as f:
            f.write("Toto je první řádek.\nToto je pokračování prvního řádku.\n\nToto je druhý, samostatný odstavec.")
        print(f"Vytvořil jsem ukázkový soubor v '{os.path.join(INPUT_DIR, 'ukazka.txt')}'")
        return

    print(f"Zpracovávám soubory z adresáře: {INPUT_DIR}")
    
    # Použití glob k nalezení všech .txt souborů
    file_paths = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    
    if not file_paths:
        print("Nebyly nalezeny žádné .txt soubory ke zpracování.")
        return

    # Otevření výstupního souboru pro zápis
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for path in file_paths:
            print(f" - Zpracovávám {os.path.basename(path)}")
            with open(path, "r", encoding="utf-8") as infile:
                # Načtení celého obsahu souboru
                content = infile.read()
                
                # Rozdělení na "dokumenty" nebo "odstavce" pomocí prázdných řádků
                # Dva nebo více prázdných řádků považujeme za oddělovač.
                # To je robustní metoda pro zpracování knih.
                paragraphs = content.split('\n\n')
                
                for para in paragraphs:
                    # Nahrazení jednoduchých konců řádků mezerou
                    # Tím se spojí řádky uvnitř odstavce
                    line = para.replace('\n', ' ').strip()
                    
                    # Odstranění nadbytečných mezer
                    line = ' '.join(line.split())
                    
                    # Zápis pouze neprázdných řádků do finálního souboru
                    if line:
                        outfile.write(line + '\n')
                        
    print(f"\nHotovo. Předzpracovaná data byla uložena do souboru: {OUTPUT_FILE}")
    
if __name__ == "__main__":
    main()