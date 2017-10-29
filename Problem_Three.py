def histogram(elems):
	elements = list(set(elems))
	histo = {}
	for number in elements: 
		histo[number] = elems.count(number)
	print(histo)
	
histogram([4, 7, 8, 7])

#histogram(elems)
	#where elems is a list
		#list —> [a,b,a,c,5]
		#output —> {a:2, b:1, c:1, 5:1}
