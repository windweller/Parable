#!/usr/bin/env bash

# This gets part of the IMDB data
# in only 2 labels: pos vs. neg
# source: CNN_sentence by Yoon Kim

wget https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.neg
wget https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.pos