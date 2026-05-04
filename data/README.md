# About the datasets

## 1. german.json
Is a single key-value pair dictionary, where the value is a list holding dictionaries of 3 key-value pairs:
- "tag" identifies the theme of the message input the chatbot receives. Examples include "Ordering", "Introduction" etc.
- "patterns" are a series of input phrases or words the chatbot should anticipate associated with the tag.
- "responses" are a series of outputs the chatbot can choose from at random to output. These are **not** static responses.

## 2. data.tsv
The dataset is a personal compiled list containing 65 basic pairs of English and German sentences. It is used to test if the Encoder-Decoder model in `Seq2Seq_1.ipynb` is able to work.

## 3. train.jsonl, val.jsonl, test.jsonl
These datasets are taken from [the Multi30k dataset](https://huggingface.co/datasets/bentrevett/multi30k), which are used in `Seq2Seq_2.ipynb`.
