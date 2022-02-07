import os
import matplotlib.pyplot as plt
import utils

## ======================================================================================##
## Neural Machine Translation ##

utils.DATA_HUB["fra-eng"] = (
    utils.DATA_URL + "fra-eng.zip",
    "94646ad1522d915e7b0f9296181140edcf86a4f5",
)


def read_data_nmt():
    """Loads the English-French dataset."""
    data_dir = utils.download_extract("fra-eng")
    with open(os.path.join(data_dir, "fra-eng/fra.txt"), "r") as f:
        return f.read()


## Preprocessing the NMT data
def no_space(char, prev_char):
    return char in set(".,!?") and prev_char != " "


def preprocess_nmt(text):
    """
    Preprocess the English-French dataaset.
    step-1. Replace non-breaking space with space.
    step-2. Convert uppercase letters to lowercase letters.
    step-3. Insert space between words and punctuation marks.
    """

    text = text.replace("\u202f", " ").replace("\xa0", " ")  ## step-1
    text = text.lower()  ## step-2
    ## step-3
    out = [
        " " + char if i > 0 and no_space(char, text[i - 1]) else char
        for i, char in enumerate(text)
    ]
    return "".join(out)


def tokenize_nmt(text, num_examples=None):
    """Tokenizes the english-french text."""
    source = []
    target = []
    for i, line in enumerate(text.split("\n")):
        if num_examples and i > num_examples:
            break
        parts = line.split("\t")
        if len(parts) == 2:
            source.append(parts[0].split(" "))
            target.append(parts[1].split(" "))
    return source, target


def show_list_len_pair_hist_nmt(legend, xlabel, ylabel, source, target):
    ## the number of tokens per sequence in source/target text
    src_count = [len(l) for l in source]
    trg_count = [len(l) for l in target]

    _, _, patches = plt.hist([src_count, trg_count])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch("/")
    plt.legend(legend)
    plt.show()


def truncate_pad(line, time_steps, padding_token):
    """
    Truncates or pads the line,
    based on whetehr its length is more or less than time_steps.
    """
    if len(line) > time_steps:
        return line[:time_steps]  ## truncation
    return line + [padding_token] * (time_steps - len(line))  ## padding


## ======================================================================================##
