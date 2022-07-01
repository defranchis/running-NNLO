def samesign (a,b,c):
    if a*b < 0: return False
    if a*c < 0: return False
    if b*c < 0: return False
    return True


def readAllXsecVsMass(filename):
    bin_low = []
    f = open(filename)
    lines = f.read().splitlines()
    for l in lines:
        if l == '': continue
        if l.replace(' ','').startswith('#'): continue
        low = int(l.split()[0])
        if not low in bin_low:
            bin_low.append(low)
    bin_low.sort()
    for i, low in enumerate(bin_low):
        print()
        for l in lines:
            if l == '': continue
            if l.replace(' ','').startswith('#'): continue
            if int(l.split()[0]) != low: continue
            central = float(l.split()[5])
            downdown = float(l.split()[2]) / central
            downnom = float(l.split()[3]) / central
            upnom = float(l.split()[4]) / central
            nomup = float(l.split()[6]) / central
            nomdown = float(l.split()[7]) / central
            upup = float(l.split()[8]) / central
            mass = float(l.split()[9])

            # print (i+1,mass,(nomup*upnom/upup-1)*100)
            # print (i+1,mass,(nomdown*downnom/downdown-1)*100)
            
            # print (i+1, mass, samesign(downdown,downnom,nomdown),samesign(upup,upnom,nomup))
            # if not samesign(downdown,downnom,nomdown) or not samesign(upup,upnom,nomup):
            #     print(downdown,downnom,nomdown)
            #     print (upup,upnom,nomup)

            # rho_up = (upup**2-upnom**2-nomup**2)/(2*nomup*upnom)
            # rho_down = (downdown**2-downnom**2-nomdown**2)/(2*nomdown*downnom)
            # print(rho_up,rho_down)

            print (i,mass)
            print (upup, upnom, nomup)
            print (downdown, downnom, nomdown)
            print()
            
    return

readAllXsecVsMass('scales_all.dat')
