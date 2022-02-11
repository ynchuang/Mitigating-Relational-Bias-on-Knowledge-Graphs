import numpy as np
fair_list = []
test_fair_name = []

with open("test.txt", 'r') as fw:
    for line in fw:
        line = line.rstrip().split('\t')
        if line[1] == '/people/person/gender':
            test_fair_name.append(line[0])
        if line[1] == '/people/person/profession':
            test_fair_name.append(line[0])

with open("gen2prof_fair_all", 'r') as fw:
    for line in fw:
        line = line.rstrip().split('\t')
        if str(line[2]) in test_fair_name:
            fair_list.append("%s\t%s\t%s\t%s\t%s" %(line[0], line[1], line[2], line[3], line[4]))


with open("gen2prof_fair_test", 'w') as fw:
    fw.write('%s\n' % ('\n'.join(fair_list)))
