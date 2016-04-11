#!/usr/bin/env bash

# This gets part of the IMDB data
# in only 2 labels: pos vs. neg
# source: CNN_sentence by Yoon Kim

wget https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.neg
wget https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.pos
mv rt-polarity.neg ./imdb_partial/rt-polarity.neg
mv rt-polarity.pos ./imdb_partial/rt-polarity.pos
rm rt-polarity.neg
rm rt-polarity.pos