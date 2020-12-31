import questions as q

s = {
	'hola1' : (15, 3),
	'hola2' : (15, 4),
	'hola3' : (20, 1)
}


sorted_s = sorted(s, key=lambda k: (s[k][0], s[k][1]), reverse=True)

print(sorted_s)
