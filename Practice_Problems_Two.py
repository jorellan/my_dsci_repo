def my_max(elements):
	elements.sort()
	print(elements[-1])
		
x = [3,7,2,10,8,9]
my_max(x)		

def word_count(filepath, case_sensitive=True):
	file = open(filepath, "r")
	word_list = file.read().split() #breaks down the words into a list
	unique_words = list(set(word_list)) 
	lower_unique = []
	words_lower = []
	wordcount={}
	if case_sensitive == True:
		for word in unique_words: 
			wordcount[word] = word_list.count(word)
		print(wordcount)
	if case_sensitive == False:
		for word in unique_words:
			lower_unique.append(word.lower())
		for low in word_list:
			words_lower.append(low.lower())
		for word in lower_unique: 
			wordcount[word] = words_lower.count(word)
		print(wordcount)


word_count("text.txt")
word_count("text.txt", False)