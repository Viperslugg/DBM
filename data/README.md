# About the datasets

## 1. german.json
Is a single key-value pair dictionary, where the value is a list holding dictionaries of 3 key-value pairs:
- "tag" identifies the theme of the message input the chatbot receives. Examples include "Ordering", "Introduction" etc.
- "patterns" are a series of input phrases or words the chatbot should anticipate associated with the tag.
- "responses" are a series of outputs the chatbot can choose from at random to output. These are **not** static responses.
