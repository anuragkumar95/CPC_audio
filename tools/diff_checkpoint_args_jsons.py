import json
import sys


a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))

keys = sorted(set(list(a.keys()) + list(b.keys())))
for k in keys:

    if k not in a:
        print(k)
        print(f'> {b[k]}')

    elif k not in b:
        print(k)
        print(f'< {a[k]}')

    else:
        # assert type(a[k]) is str, f'{k} {type(a[k])} {a[k]}'
        # assert type(b[k]) is str, f'{k} {type(b[k])} {b[k]}'

        if a[k] != b[k]:
            print(k)
            print(f'< {a[k]}')
            print(f'> {b[k]}')
