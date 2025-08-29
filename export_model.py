# export_model.py
import torch
import os
import argparse
import sentencepiece as spm

import config
from model import Transformer
from plant_net import PlantLayer

def main():
    parser = argparse.ArgumentParser(description="Exports the complete state model from the training checkpoint.")
    parser.add_argument('--input', type=str, default=config.generator_checkpoint_path, help="Path to the input checkpoint file (.pt)")
    parser.add_argument('--output', type=str, default="finetuned_model.pth", help="Path to the output file with the complete model (.pth)")
    args = parser.parse_args()

    print(f"Loading checkpoint from: {args.input}")
    if not os.path.exists(args.input):
        print(f"ERROR: Input file '{args.input}' does not exist.")
        return

    checkpoint = torch.load(args.input, map_location='cpu')
    
    sp = spm.SentencePieceProcessor()
    sp.load(f'{config.model_prefix}.model')
    actual_vocab_size = sp.get_piece_size()

    print("I am creating the complete model architecture for export...")
    model = Transformer(
        vocab_size=actual_vocab_size,
        dim=config.embedding_dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        ff_multiplier=config.ff_dim_multiplier,
        dropout=0.0
    )
    
    state_dict = checkpoint['model_state_dict']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    print("LLM weights were successfully loaded and cleaned.")

    initial_plant_state = None
    if model.use_plant_net:
        print("Generating default state for the plant network...")
        clean_plant_layer = PlantLayer(state_file=None)
        initial_plant_state = {
            "width": clean_plant_layer.network.width,
            "height": clean_plant_layer.network.height,
            "cells": [[c.to_dict() for c in row] for row in clean_plant_layer.network.cells]
        }

    exported_data = {
        'model_state_dict': model.state_dict(),
        'initial_plant_state': initial_plant_state, 
        'config': {
            'vocab_size': actual_vocab_size,
            'embedding_dim': config.embedding_dim,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'n_kv_heads': config.n_kv_heads,
            'ff_multiplier': config.ff_dim_multiplier,
            'pretrain_max_seq_len': config.pretrain_max_seq_len,
            'use_plant_network': model.use_plant_net
        }
    }
    
    torch.save(exported_data, args.output)
    print(f"Kompletní model byl úspěšně exportován do: {args.output}")
    print(f"Velikost souboru: {os.path.getsize(args.output) / 1e6:.2f} MB")

if __name__ == '__main__':
    main()