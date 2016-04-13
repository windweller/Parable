#!/usr/bin/env bash

mkdir story_corpus_16
wget http://web.stanford.edu/~anie/download/ROCStories__spring2016%20-%20ROC-Stories-naacl-camera-ready.csv
wget http://web.stanford.edu/~anie/download/cloze_test_test__spring2016%20-%20cloze_test_ALL_test.csv
wget http://web.stanford.edu/~anie/download/cloze_test_val__spring2016%20-%20cloze_test_ALL_val.csv

mv ROCStories__spring2016\ -\ ROC-Stories-naacl-camera-ready.csv story_corpus_16
mv cloze_test_val__spring2016\ -\ cloze_test_ALL_val.csv story_corpus_16
mv cloze_test_test__spring2016\ -\ cloze_test_ALL_test.csv story_corpus_16