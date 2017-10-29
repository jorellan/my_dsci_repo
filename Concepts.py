increment = lambda x: x + 1
print(increment(3))

squares_upto = lambda n: map(lambda x: x * x, range(1, n + 1))
print(squares_upto(5))

hi = reduce(lambda x,y: x * y, [1, 2, 3, 4, 5])
print(hi)

print(map(5,6))