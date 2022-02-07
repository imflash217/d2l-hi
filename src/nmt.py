import utils
import utils_nmt


## ======================================================================================##
## Neural Machine Translation ##

## Step-1: Data Loading
raw_text = utils_nmt.read_data_nmt()
print(raw_text[:75])
print("---" * 20)

## step-2: Preprocess the raw data
text = utils_nmt.preprocess_nmt(raw_text)
print(text[:75])
print("---" * 20)

## step-3: Tokenization
source, target = utils_nmt.tokenize_nmt(text)
print(source[:5])
print("~~" * 20)
print(target[:5])

## plotting the density histogram
utils_nmt.show_list_len_pair_hist_nmt(
    ["source", "target"], "#tokens per sequence", "# of sequences", source, target
)

## step-4: Vocabulary
src_vocab = utils.Vocab(source, min_frequency=2, reserved_tokens=["<pad>", "<bos>", "<eos>"])
trg_vocab = utils.Vocab(target, min_frequency=2, reserved_tokens=["<pad>", "<bos>", "<eos>"])

## ======================================================================================##
