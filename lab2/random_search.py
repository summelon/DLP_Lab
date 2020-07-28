import re
import sys
import subprocess

while(1):
    outputs = subprocess.check_output("python train.py --model_name eegnet --mode train | grep -P \"val_acc\" | awk '{print $NF}'", shell=True)
    results = re.findall(r'[0-9\.]+', outputs.decode())
    print(results)
    if any([float(number) > 87 for number in results]):
        sys.exit()
