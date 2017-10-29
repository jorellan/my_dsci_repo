#Assignment One ~ Jennifer Orellana 
#def flatten(x):
def product(*seqs):
    if not seqs:
        return [[]]
    else:
        return [[x] + p for x in seqs[0] for p in product(*seqs[1:])]

hi = product([5,4,3], [2,4,6], [10, 11, 12])
#print(hi)


def c_p(*info):
	if not info:
		return[[]]
	else:
		#for p in c_p(*info[1:]):
			#return [p]
		return c_p(*info)
		return [[x] for x in info[0]]
hii = c_p([5,4,3], [2,4,6])
print(hii)