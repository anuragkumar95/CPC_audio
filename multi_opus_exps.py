import subprocess
from subprocess import TimeoutExpired

availableMachines = ['opus0', 'opus1', 'opus2']

seqCheckpoints = [
    '/pio/scratch/1/i323106/wav2vec/runs/cpc/hacpc-gt-cosine-norelu-encodeseg-ls/checkpoint_49.pt',
    '/pio/scratch/1/i323106/wav2vec/runs/cpc/hacpc1neg2-6-12-2-4-cosine-norelu-kreuk-encodeseg-ls+buckeye/checkpoint_49.pt', # 8 not 1 neg
    '/pio/scratch/1/i323106/wav2vec/runs/cpc/hacpc1neg2-cosine-norelu-kreuk-encodeseg-ls+buckeye/checkpoint_49.pt',
    '/pio/scratch/1/i323106/wav2vec/runs/cpc/hacpc1neg-6-12-2-4-gt-cosine-norelu-encodeseg-ls/checkpoint_49.pt',    # 8 not 1 neg
    '/pio/scratch/1/i323106/wav2vec/runs/cpc/hacpc8neg-6-12-1-1-gt-cosine-norelu-encodeseg-ls/checkpoint_49.pt',
    '/pio/scratch/1/i323106/wav2vec/runs/cpc/hacpc32neg-6-12-1-1-gt-cosine-norelu-encodeseg-ls/checkpoint_49.pt',
]
seqCommands = []
for i, chkpt in enumerate(seqCheckpoints):
    seqCommands += [
        f'bash lineval_ls100.sh {chkpt} --CTC --CTC_forbid_blank --useLSTM --convClassifier --ignore_cache',
        f'bash lineval_ls100.sh {chkpt} --get_encoded --CTC --CTC_forbid_blank --useLSTM --convClassifier --ignore_cache',
        f'bash lineval_ls100.sh {chkpt} --CPCLevel 1 --CTC --CTC_forbid_blank --useLSTM --linearClassifier --ignore_cache --upsampleSeq',
        f'bash lineval_ls100.sh {chkpt} --CPCLevel 1 --get_encoded --CTC --CTC_forbid_blank --useLSTM --linearClassifier --ignore_cache --upsampleSeq',
        f'bash phoneseg_buckeye.sh {chkpt} --boundaryDetector kreuk --get_encoded',
        f'bash wordseg_buckeye.sh {chkpt} --boundaryDetector kreuk'
    ]
sshProcesses = []
print(f"Running {len(seqCommands)} commands")
while len(seqCommands) > 0:
    while len(availableMachines) == 0:
        for i, (machine, cmd, process) in enumerate(sshProcesses):
            try:
                process.communicate(timeout=1)
                if process.returncode != 0:
                    print(f"\tD: Experiment with command {cmd} on machine {machine} failed with status {process.returncode}.")
                else:
                    print(f"\tExperiment with command {cmd} on machine {machine} finished successfully :D")
                availableMachines.append(machine)
                sshProcesses.pop(i)
                break
            except TimeoutExpired:
                pass
                
    machine = availableMachines.pop()
    expCmd = seqCommands.pop()

    cmd = f'ssh {machine} "source /pio/scratch/1/i323106/miniconda3/bin/activate;conda activate cpc37;cd /pio/scratch/1/i323106/CPC_audio;'
    expCmd = expCmd + '"'
    cmd = cmd + expCmd
    print(f"Executing new experiment on {machine} with command :\n\t{expCmd[:-1]}")

    sshProcesses.append(
        (
            machine,
            expCmd[:-1],
            subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        )
    )

while len(sshProcesses) > 0:
    for i, (machine, cmd, process) in enumerate(sshProcesses):
        try:
            process.communicate(timeout=1)
            if process.returncode != 0:
                print(f"\tD: Experiment with command {cmd} on machine {machine} failed with status {process.returncode}.")
            else:
                print(f"\tExperiment with command {cmd} on machine {machine} finished successfully :D")
            availableMachines.append(machine)
            sshProcesses.pop(i)
            break
        except TimeoutExpired:
            pass
