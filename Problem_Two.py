def print_triangles(width, full=False):
	if full==False:
		for stars in range(1, width+1):
			triangle = "*" * stars
			print(triangle)
	if full==True:
		for stars in range(1, width+1):
			triangle = "*" * stars
			print(triangle)
		for stars in reversed(range(1, width+1)):
			triangle = "*" * stars
			print(triangle)
			
print_triangles(8, True)
print("\n")
print_triangles(5)

#print_triangle(width, full = false)
#print triangle
 	# * 
	# **
	# ***
#full —> print the whole triangle 
