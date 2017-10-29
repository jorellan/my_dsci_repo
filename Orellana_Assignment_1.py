#Assignment One ~ Jennifer Orellana 

#Flatten 
def flatten(information):
	q = []
	for info in information:
		if isinstance(info, list) == False: 
			q.append(info)
		else:
			q = q + flatten(info)
	return q
			
#Power Set
def powerset(information):
	beg = [[]]
	for info in information:
		beg.extend([start + [info] for start in beg])
	return (beg)
	
#All Permutations 
def all_perms(info):
	if len(info) == 0:
		return []
	elif len(info) == 1:
		return [info]
	else:
		result = []
		for i in range(len(info)):
			add = info[i]
			rest = info[:i] + info[i+1:]
			for each in all_perms(rest):
				result.append([add] + each)
		return result

#Number Spiral
def left(num,d):
	d.insert(0,num)
	return d

def right(a,d):
	d.append(a)
	return d

def middle(num,d):
	d.insert((len(d)/2),num)
	return d
	
def flatten(information):
	q = []
	for info in information:
		if isinstance(info, list) == False: 
			q.append(info)
		else:
			q = q + flatten(info)
	return q
		
def four(num):
	lst = [[] for _ in range(num)]
	cn = (num**2-1)
	lst[-1].append(cn)
	for x in range(num-2,-1,-1):
		lst[x].append(cn-1)
		cn = cn-1
	for x in range(num-1):
		left(cn-1,lst[0])
		cn = cn-1
	for x in range(1, num):
		lst[x].insert(0,(cn-1))
		cn = cn-1
	for x in range(num-2):
		lst[-1].insert(0,(cn-1))
		cn = cn-1
	t = lst[-1][:-1]
	t.reverse()
	t.append(lst[-1][-1:])
	lst[-1] = t
	return lst


def two(num):
	lst = [[] for _ in range(num)]
	cn = (num**2-1)
	lst[0].append(cn)
	for x in range(num-1):
		left(cn-1,lst[0])
		cn = cn-1
	for x in range(1, num):
		lst[x].append(cn-1)
		cn = cn-1
	for x in range(num-1):
		right(cn-1,lst[-1])
		cn = cn-1
	for x in range(num-2,0,-1):
		lst[x].append(cn-1)
		cn = cn-1
	if num-2 != 0:
		u = []
		for x in range(num-2):
			u.insert(0,cn-1)
			cn = cn -1
		middle(u,lst[1])
	return lst

def one(num):
	lst = [[] for _ in range(num)]
	cn = (num**2-1)
	lst[0].append(cn)
	for x in range(1, num):
		lst[x].append(cn-1)
		cn = cn-1
	for x in range(num-1):
		right(cn-1,lst[-1])
		cn = cn-1
	for x in range(num-2,0,-1):
		lst[x].append(cn-1)
		cn = cn-1
	for x in range(num-1):
		left(cn-1,lst[0])
		cn = cn-1
	t = lst[0][:-1]
	t.insert(0,(lst[0][-1:]))
	lst[0] = t
	return lst
	
def three(num):
	lst = [[] for _ in range(num)]
	cn = (num**2-1)
	lst[-1].append(cn)
	for x in range(num-1):
		right(cn-1,lst[-1])
		cn = cn-1
	for x in range(num-2,-1,-1):
		lst[x].append(cn-1)
		cn = cn-1
	for x in range(num-1):
		left(cn-1,lst[0])
		cn = cn-1
	for x in range(1, num-1):
		lst[x].insert(0,(cn-1))
		cn = cn-1
	return lst
	
def spiral(n,corner):
	if corner == 1:
		numbers = one(n)
	elif corner == 2:
		numbers = two (n)
	elif corner == 3:
		numbers = three(n)
	elif corner == 4:
		numbers = four(n)
	for x in numbers:
		x = flatten(x)
		print x
                                 ## #4 Incomplete
