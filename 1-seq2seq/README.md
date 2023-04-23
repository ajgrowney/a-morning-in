# Goal
Use python to generate n words 

# Setup
Got lyrics data from: https://github.com/fpaupier/RapLyrics-Scraper

# What Happened
From initial results, I couldn't find a many-to-many model example, but this tensorflow example notebook did a really good job at explaining many-to-one.


# What I learned
1. Sequence to sequence models are actually a sub-class of RNNs
2. Keeping state can be done in memory which is nice and gives me good insight on creating "sessions"
3. "Encoding" in this case is just character -> id
4. "Decoding" in this case is just id -> character
5. There is an encoding schema called "Sentence Piece" that Google uses and is commonly adopted in production systems that is above the character level, but below the word level
6. GPT uses "tiktoken" for encoding (has 50,257 vocab size)