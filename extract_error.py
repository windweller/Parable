"""
This file extract errors and generate a new file
"""
import os
import re

if __name__ == '__main__':
    pwd = os.path.dirname(os.path.realpath(__file__))
    with open(pwd + '/sunday_night_n_19_resfuse_epoch_90.log', mode='r') as f:
        training_loss = []
        valid_loss = []
        valid_acc = []
        for line in f:
            if 'top 5 validation accuracy' in line:
                number = re.findall(r'[-+]?\d+[\.]?\d*', line.split('top 5 validation accuracy')[1])
                print 100 - float(number[0])
