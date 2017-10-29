def my_range(a, b, x=1):
	numbers = [a]
	for number in range (a,b):
		while (a<b-x):
			a = a+x
			numbers.append(a)
	print(numbers)
		
my_range(2,10,2)
my_range(30, 100, 10)

#my_range (a, b, by = 1)
	#starts from first number and doesn’t include last number 
	#use for loops



