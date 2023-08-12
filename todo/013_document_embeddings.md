# 013 Document Embeddings

- TODO
  - [ ] cache embeddings with model name to allow user to experiment with different embedding models easily.
  - [ ] use spacy instead of sliding window to do embeddings. This will necessitate changes to caching.
  - [ ] option to recursively search a directory and find relevant docs, then relevant sections.
  - [ ] consider token limit of models
  - [ ] For HTML, we should render out html tags, boil it all down to just text
  - [ ] Lessons to learn from langchain?
  - [ ] Embedding models are noisy AF. Can we, say, data-augment a query by generating permutations of it, then average the query, or similarity results for gain?

- Support
  - [X] Github Repos
  - [X] PDFs
  - [X] Individual docs
  - [X] Local harddrive files
  - [X] HTML
