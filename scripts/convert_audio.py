import os
import subprocess
from tqdm import tqdm
import argparse
from multiprocessing import Pool, Manager, Process, RLock

def convert_audio(fin, fout):
    """
    Reads the file from fin and saves the file in any format in fout
    """
    temp = subprocess.run(["ffmpeg",
                           "-i", 
                           fin, 
                           fout], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE)


def run_function(pid, files, args, start, end): 
    """
    Function to be run on different processes.
    """
    tqdm_text = "#"+"{}".format(pid).zfill(3)
    with tqdm(total=end-start, desc=tqdm_text, position=pid+1) as pbar:
        for audio in files[start:end]:
            audio_name, extn = audio.split('.')
            print(audio, audio_name, extn)
            if extn != 'm4a':
                continue 
            if audio_name+'.'+args.extn in os.listdir(args.res_dir):
                continue
            inp_audio = args.data_dir + "/" + audio
            out_audio = args.res_dir + "/" + audio_name + "." + args.extn
            #convert_audio(inp_audio, out_audio)
        pbar.update(1)

def main(args):
    files = os.listdir(args.res_dir)
    pool = Pool(processes=args.num_process, initargs=(RLock(), ), initializer=tqdm.set_lock)
    processes = []
    num_files = len(files)
    files_per_process = int(num_files/args.num_process) + 1
    print('\n')
    for i in range(args.num_process):
        start = int(i*files_per_process)
        end = int(min(start + files_per_process, num_files))
        processes.append(pool.apply_async(run_function, args=(i, 
                            files, 
                            args, 
                            start, 
                            end,)))
    pool.close()
    results = [job.get() for job in processes]



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_process", type=int, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--res_dir', type=str, required=True, help='Path to dir where the results will be stored.')
    parser.add_argument('--extn', type=str, required=False, help='extn of output format')
    args = parser.parse_args()
    main(args)