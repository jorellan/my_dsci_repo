#1
def my_range(a, b, x=1):
	numbers = [a]
	for number in range (a,b):
		while (a<b-x):
			a = a+x
			numbers.append(a)
	print(numbers)
		
my_range(2,10)
my_range(30, 100, 10)
print("\n")

#2
def print_triangles(width, full=False):
	for stars in range(1, width+1):
		triangle = "*" * stars
		print(triangle)
	if full==True:
		for stars in reversed(range(1, width+1)):
			triangle = "*" * stars
			print(triangle)
			
print_triangles(4, True)
print("\n")
print_triangles(8)
print("\n")

#3
def histogram(elems):
	elements = list(set(elems))
	histo = {}
	for number in elements: 
		histo[number] = elems.count(number)
	print(histo)
	
histogram([4, 7, 8, 7])
histogram(["a","a","a","a","a","a","b","b","b","b","c","c","c","d", "d", "e"])

