from questions import load_files, tokenize

files = load_files("corpus")

example = "(and by 1959 were reportedly playing better than the average human)"
print(tokenize(example))