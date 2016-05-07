import os
import string
from os.path import join as pjoin

# reference: https://github.com/mila-udem/blocks-examples/blob/master/machine_translation/configurations.py

'''
path settings
'''

UG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ['DATA_DIR']
assert DATA_DIR != ''
UG_DATA_DIR = pjoin(UG_DIR, 'data')

MT_DATA_PATH = pjoin(DATA_DIR, 'wmt')
NLC_DATA_PATH = pjoin(DATA_DIR, 'nlc')
LANG8_DATA_PATH = pjoin(NLC_DATA_PATH, 'lang8/lang-8-en-1.0')

'''
character models
'''

CHAR_SOS_TOK = '<sos>'
CHAR_EOS_TOK = '<eos>'
CHAR_UNK_TOK = '<unk>'
CHAR_MAX_SEQ_LEN = 200
CHAR_SORT_K_BATCHES = 12
# reduce batch sizes if using attention to avoid memory error
CLIP_RATIO = 2
# used for sanity checking
CHAR_FILE = pjoin(UG_DATA_DIR, 'tinyshakespeare.txt')
PTB_FILE = pjoin(UG_DATA_DIR, 'ptb.txt')

'''
word mt models
'''

SRC = 'en'
TGT = 'de'
# NOTE preprocess.py will produce dictionary of size VOCAB_SIZE + 1 since <s> and </s> mapped to same index
SRC_VOCAB_SIZE = 50000
TGT_VOCAB_SIZE = 50000
LANG8_VOCAB_SIZE = 50000

WORD_EOS_TOK = '</s>'
WORD_UNK_TOK = 'UNK'
EOS_IND = 0  # must agree w/ preprocess.py
UNK_IND = 1
WORD_MAX_SEQ_LEN = 50
# tried sorting 100 batches but hurt performance on error correction dataset, may vary with dataset
WORD_SORT_K_BATCHES = 12

'''
decoding
'''

DECODER_MAX_LENGTH_RATIO = 1.5
DECODER_MIN_LENGTH_RATIO = 1.5
DECODER_LENGTH_ABS_DIFF = 5
DECODER_MAX_EDIT_DIST_RATIO = 0.5

'''
third-party
'''

# useful scripts at https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
TOKENIZER = pjoin(UG_DIR, 'scripts/mt/tokenizer.perl')
# TODO use CoreNLP detokenizer instead of this
DETOKENIZER = pjoin(UG_DIR, 'scripts/mt/detokenizer.perl')
PREPROCESSOR = pjoin(UG_DIR, 'scripts/mt/preprocess.py')
BLEU_SCORER = pjoin(UG_DIR, 'scripts/mt/multi-bleu.perl')
MT_SCRIPT_DIR = pjoin(UG_DIR, 'scripts/mt')
MT_PREFIX_DIR = pjoin(UG_DIR, 'scripts/share/nonbreaking_prefixes')
M2_SCORER = pjoin(NLC_DATA_PATH, 'm2scorer/m2scorer')

# Stanford CoreNLP
CORENLP_PATH = '/afs/cs.stanford.edu/u/zxie/libs/stanford-corenlp-full-2014-08-27'

'''
machine translation paths and datasets
'''

TRAIN_CORPUSES = ['commoncrawl', 'europarl-v7', 'news-commentary-v9']
#TRAIN_CORPUSES = ['commoncrawl', 'europarl-v7', 'news-commentary-v10']
VALID_CORPUS_FILES = [
        pjoin(MT_DATA_PATH, 'dev/newstest2013.en'),
        pjoin(MT_DATA_PATH, 'dev/newstest2013.de'),
]
# XXX need to create these files from the sgm files
TEST_CORPUS_SGM_FILES = [
        pjoin(MT_DATA_PATH, 'test/newstest2014-deen-src.en.sgm'),
        #pjoin(MT_DATA_PATH, 'test/newstest2015-ende-src.en.sgm'),
        pjoin(MT_DATA_PATH, 'test/newstest2014-deen-ref.de.sgm'),
        #pjoin(MT_DATA_PATH, 'test/newstest2015-ende-ref.de.sgm'),
]
TEST_CORPUS_FILES = [
        pjoin(MT_DATA_PATH, 'test/newstest14.en'),
        pjoin(MT_DATA_PATH, 'test/newstest14.de'),
]

SRC_DICT_FILE = pjoin(
    MT_DATA_PATH, 'vocab.{}-{}.{}.pkl'.format(SRC, TGT, SRC))
TGT_DICT_FILE = pjoin(
    MT_DATA_PATH, 'vocab.{}-{}.{}.pkl'.format(SRC, TGT, TGT))
LANG8_DICT_FILE = pjoin(LANG8_DATA_PATH, 'vocab.pk')

SRC_TRAIN_FILE = pjoin(
    MT_DATA_PATH, 'train.%s-%s.%s.tok.shuf' % (SRC, TGT, SRC))
TGT_TRAIN_FILE = pjoin(
    MT_DATA_PATH, 'train.%s-%s.%s.tok.shuf' % (SRC, TGT, TGT))
SRC_VALID_FILE = pjoin(
    MT_DATA_PATH, 'newstest2013.%s.tok' % SRC)
TGT_VALID_FILE = pjoin(
    MT_DATA_PATH, 'newstest2013.%s.tok' % TGT)
SRC_TEST_FILE = pjoin(
    MT_DATA_PATH, 'newstest14.%s.tok' % SRC)
TGT_TEST_FILE = pjoin(
    MT_DATA_PATH, 'newstest14.%s.tok' % TGT)

'''
language correction paths and datasets

nucle data obtained from:
    - http://www.comp.nus.edu.sg/~nlp/conll13st.html
    - http://www.comp.nus.edu.sg/~nlp/conll14st.html
    - version 3.2 requested from organizers

lang-8 data requested from http://cl.naist.jp/nldata/lang-8/
'''

# NOTE already shuffled in processs_lang8.py
#ORIG_TRAIN_FILE = pjoin(LANG8_DATA_PATH, 'entries.train.original')
#CORR_TRAIN_FILE = pjoin(LANG8_DATA_PATH, 'entries.train.corrected')
#NUCLE_ORIG_TRAIN_FILE = '/deep/group/zxie/nlc/nucle3.2/data/train_original.txt'
#NUCLE_CORR_TRAIN_FILE = '/deep/group/zxie/nlc/nucle3.2/data/train_corrected.txt'
# combine lang8 and nucle data
#ORIG_TRAIN_FILE = '/deep/group/zxie/nlc/lang8+nucle+aug.train.original.shuf'
#CORR_TRAIN_FILE = '/deep/group/zxie/nlc/lang8+nucle+aug.train.corrected.shuf'
# combine lang8 and nucle data + augmented nucle training
ORIG_TRAIN_FILE = '/deep/group/zxie/nlc/lang8+nucle+fce+aug2.train.original.shuf'
CORR_TRAIN_FILE = '/deep/group/zxie/nlc/lang8+nucle+fce+aug2.train.corrected.shuf'
ORIG_VALID_FILE = pjoin(LANG8_DATA_PATH, 'entries.dev.original')
CORR_VALID_FILE = pjoin(LANG8_DATA_PATH, 'entries.dev.corrected')
ORIG_TEST_FILE = pjoin(LANG8_DATA_PATH, 'entries.test.original')
CORR_TEST_FILE = pjoin(LANG8_DATA_PATH, 'entries.test.corrected')
# these files are from when tried to generate lang8 english corpus myself...
#ORIG_TRAIN_FILE = pjoin(LANG8_DATA_PATH, 'new.train.original')
#CORR_TRAIN_FILE = pjoin(LANG8_DATA_PATH, 'new.train.corrected')
#ORIG_VALID_FILE = pjoin(LANG8_DATA_PATH, 'new.dev.original')
#CORR_VALID_FILE = pjoin(LANG8_DATA_PATH, 'new.dev.corrected')
#ORIG_TEST_FILE = pjoin(LANG8_DATA_PATH, 'new.test.original')
#CORR_TEST_FILE = pjoin(LANG8_DATA_PATH, 'new.test.corrected')

CONLL_2013_TEST_M2_FILE = pjoin(NLC_DATA_PATH, 'nucle2.3.1/revised/data/official-preprocessed.m2')
#CONLL_2013_TEST_M2_FILE = pjoin(NLC_DATA_PATH, 'nucle3.2/data/conll14st-preprocessed.m2')
#CONLL_2013_TEST_M2_FILE = pjoin(NLC_DATA_PATH, 'nucle2.3.1/revised/data_5types/official-preprocessed.5types.m2')
#CONLL_2013_TEST_M2_FILE = pjoin(NLC_DATA_PATH, 'nucle3.2/data/dev.m2')
CONLL_2014_TEST_M2_FILE = pjoin(NLC_DATA_PATH, 'conll14st-test-data/noalt/official-2014.combined.m2')
CONLL_2014_TEST_M2_FILE_ANNOT1 = pjoin(NLC_DATA_PATH, 'conll14st-test-data/noalt/official-2014.0.m2')
CONLL_2014_TEST_M2_FILE_ANNOT2 = pjoin(NLC_DATA_PATH, 'conll14st-test-data/noalt/official-2014.1.m2')

# NOTE these are the uncorrected files
CONLL_2013_TEST_TXT_FILE = pjoin(NLC_DATA_PATH, 'noalt2013.txt')
# NOTE this was for generating edits for edit classification
#CONLL_2013_TEST_TXT_FILE = pjoin(NLC_DATA_PATH, 'nucle3.2/data/train_original.txt')
#CONLL_2013_TEST_TXT_FILE = pjoin(NLC_DATA_PATH, 'noalt2013.5types.txt')
CONLL_2014_TEST_TXT_FILE = pjoin(NLC_DATA_PATH, 'noalt2014.txt')

FCE_DATASET_PATH = pjoin(NLC_DATA_PATH, 'clc_fce')
