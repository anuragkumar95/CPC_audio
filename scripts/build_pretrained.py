import os
import numpy as np
import s3prl.hub as hub
import torch
import torchaudio
import progressbar
import time
import argparse

def get_model(model, device):
    return getattr(hub, model)().to(device)

def build_embeddings(model, wav_path, savepath):
    seqs = []
    for wav in os.listdir(wav_path):
        filepath = os.path.join(wav_path, wav)
        wav_seq = torchaudio.load(filepath)[0]
        seqs.append((wav_seq, wav))
    
    # Building features
    print("")
    print(f"Building CPC features and saving outputs to {savepath}...")
    bar = progressbar.ProgressBar(maxval=len(seqs))
    bar.start()
    start_time = time()
    
    with torch.no_grad():
        for i, (seq, seq_name) in enumerate(seqs):
            reps = model(seq)['hidden_states']
            file_out = os.path.join(savepath, seq_name, '.txt')
            np.savetxt(file_out, reps)
            bar.update(i)
    
    bar.finish()
    print(f"...done {len(seqs)} files in {time()-start_time} seconds.")


def main(args):
    if args.cpu:
        device = 'cpu'
    else:
        device = 'cuda' 
    model = get_model(args.model, device)
    build_embeddings(model, args.pathDB, args.save)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--save', required=True, type=str)
    parser.add_argument('--pathDB', required=True, type=str)
    



            
