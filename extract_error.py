"""
This file extract errors and generate a new file
"""
import os
import re

if __name__ == '__main__':
    pwd = os.path.dirname(os.path.realpath(__file__))
    with open(pwd + '/saturday_eve_n_34_resnet_epoch_90.log', mode='r') as f:
        training_loss = []
        valid_loss = []
        valid_acc = []
        for line in f:
            if 'training loss' in line:
                number = re.findall(r'[-+]?\d+[\.]?\d*', line)
                print number[0]
