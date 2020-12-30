import questions as q

files = q.load_files("corpus")

file_words = {
	filename: q.tokenize(files[filename])
	for filename in files
}

file_idfs = q.compute_idfs(file_words)
query = set(q.tokenize(input("Query: ")))

print(q.top_files(query, file_words, file_idfs, n=3))


