f1 = open('cleanedPosTopics', "r")
f = open('cleanedNegTopics', "r")
f_out = open("posNegWords","w")
s1 = set()
s2 = set()
spos = set()
sneg = set()

for line in f:
	l = line.strip().split(' ')
	for i in l:
		s1.add(i)

for line in f1:
	l = line.strip().split(' ')
	for i in l:
		#temp = i
		s2.add(i)
		if i not in s1:
			spos.add(i)


for i in s1:
	if i not in s2:
		sneg.add(i)
	
sneg = list(sneg)
spos = list(spos)
spos.append("great")
spos.append("good")
spos = ' '.join(spos)
sneg = ' '.join(sneg)

f_out.write(spos)
f_out.write ('\n')
f_out.write(sneg)
f_out.write ('\n')


