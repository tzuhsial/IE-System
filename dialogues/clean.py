import json
import sys

in_file = sys.argv[1]
out_file = sys.argv[2]


dialogues = []
with open(in_file, 'r') as fin:
    for line in fin.readlines():
        d = json.loads(line)
        if 'result' not in d:
            continue
        if 'turns' not in d:
            continue
        if len(d['turns']) == 0:
            continue
        if 'policy' not in d:
            continue
        dialogues.append(d)

print("filtered", len(dialogues), 'dialogues')

with open(out_file, 'w') as fout:
    for d in dialogues:
        json_line = json.dumps(d) + '\n'
        fout.write(json_line)
