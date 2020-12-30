import questions as q

files = q.load_files("corpus")

file_words = {
	filename: q.tokenize(files[filename])
	for filename in files
}

print(q.compute_idfs(file_words))
