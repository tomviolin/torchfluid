
def childname(pn):
    cn = pn
    if cn[-1].isnumeric():
        cn=cn+"_000001c"
        return cn
    sn = int(cn[-7:-1])+1
    sn=f"{sn:06d}"
    cn=cn[:-7]+sn+cn[-1]
    return cn
while True:
    print(childname(input()))
