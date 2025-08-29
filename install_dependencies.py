# install_dependencies.py
import sys
import subprocess
import platform

def install_package(package):
        try:
        print(f"--- Instaluji balíček: {package} ---")
        # Použití sys.executable zajišťuje, že se použije pip z aktuálního prostředí
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"--- Package '{package}' successfully installed. ---\n")
    except subprocess.CalledProcessError as e:
        print(f"!!! ERROR: Can not instal the package '{package}'. !!!")
        print(f"ERR msg: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("!!! ERROR: 'pip' command was not foud. Be sure, your Python and pip is installed and added to the system path (PATH).")
        sys.exit(1)

def main():
    print("==============================================================")
    print("  Hybridn LLM with PlantNet laeyers - dependencies installer")
    print("  (c)2025 OpenTechLab Jablonec nad Nisou s. r. o.
    print("==============================================================")
    print("This script will install all needed libraries.\n")

    packages = [
        "numpy",
        "sentencepiece",
        "protobuf",
        "transformers",
        "tqdm",
        "matplotlib",
        "gradio"
    ]

    print("Step 1: PyTorch installation")
    print("-------------------------")
    print("Installing basic 'torch' packet.")
    print("NOTE: For full GPU (NVIDIA CUDA) support may be needed specific command.")
    print("If you meet troubles with CUDA, visit https://pytorch.org/get-started/locally/")
    print("and use command, generatet for your CUDA version.\n")
    
    install_package("torch")

    print("\nStep 2: Installation of others dependencies")
    print("-------------------------------------")
    for package in packages:
        install_package(package)
        
    print("======================================================")
    print("  All dependencies successfully installed!")
    print("  Start your project...")
    print("======================================================")

if __name__ == "__main__":
    main()