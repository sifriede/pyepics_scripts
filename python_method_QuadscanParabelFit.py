# !/usr/bin/python

import numpy as np
import scipy.odr as odr
import matplotlib.pyplot as plt

### _____________________Plot und Fit der Parabel__________________________###
### Bei schlechten bzw. neuen Fits jeweils start und end ändern, sowie die
### richtige Fitfunktion nehmen. Danach die Startparameter startpar ändern

def fitfkt1(m, kvalue):
    s, l = m[3], m[4]
    theta = s * np.sqrt(kvalue)
    return m[0] * 1e-06 * (np.cos(theta) - l * np.sqrt(kvalue) * np.sin(theta)) ** 2 + \
        m[2] * 1e-06 * (l * np.cos(theta) + np.sqrt(1 / kvalue) * np.sin(theta)) ** 2 + \
        2 * m[1] * 1e-06 * (np.cos(theta) - l * np.sqrt(kvalue) * np.sin(theta)) * \
        (l * np.cos(theta) + np.sqrt(1 / kvalue) * np.sin(theta))

def fitfkt2(m, kvalue):
    s, l = m[3], m[4]
    theta = s * np.sqrt(kvalue)
    return m[0] * 1e-06 * (np.cosh(theta) + l * np.sqrt(kvalue) * np.sinh(theta)) ** 2 + \
        m[2] * 1e-06 * ( l * np.cosh(theta) + np.sqrt(1 / kvalue) * np.sinh(theta)) ** 2 + \
        2 * m[1] * 1e-06 * (np.cosh(theta) + l * np.sqrt(kvalue) * np.sinh(theta)) * \
        (l * np.cosh(theta) + np.sqrt(1 / kvalue) * np.sinh(theta))
        
        
def fitparabel(fitfkt, kwerte, sigsqwerte, fehler, par, seite):
    """sigsqwerte = sig in m, fehler = sigsqma_fehler in m, ifixb = variiere, wenn Parameter = 1, sonst 0"""
    # Relativistic parameter
    gamma = 1.196
    beta = 0.5482
    
    if len(kwerte) != len(sigsqwerte):
        print("Number of k-values does not match number of sigma-squared valus.")
        print("Aborting")
        return None
    
    fkt = odr.Model(fitfkt)
    data = odr.Data(kwerte, sigsqwerte, we=1. / np.power(fehler, 2))
    odrparabel = odr.ODR(data, fkt, beta0=par[0], ifixb=par[1], maxit=1000)
    odroutput = odrparabel.run()
    
    if (odroutput.beta[0] * odroutput.beta[2] - odroutput.beta[1]**2) > 0:
        eps = beta * gamma * np.sqrt(odroutput.beta[0] * odroutput.beta[2] - odroutput.beta[1] **2)
        deps = beta * gamma *np.sqrt((odroutput.beta[2]*odroutput.sd_beta[0])**2 + (odroutput.beta[0]*odroutput.sd_beta[2])**2 + (2*odroutput.beta[1]*odroutput.sd_beta[1])**2) /(2*eps)
        return eps, deps, odroutput.beta
    else:
        print("Emittanz aus ", seite, " Teil der Parabel ist imaginär.")
        return 0, 0, odroutput.beta


def parfit(k, sigsq, delsigsq, s, l):
    """sigsq = sig**2 in m**2,  delsigsq = delsig**2 in m**2"""
    
    #Fit parameter
    """ 
    Erhöhen des 1. Parameters => steileren Kurve und das Minimum wird zur 0 verschoben
    Erhöhen des 2. Parameters => Verschiebung des Minimums von der 0 weg und Verschiebung in negative y-Richtung
    Erhöhen des 3. Parameters => Verschiebung in positive y-Richtung und Minimum wird leicht weg von der 0 geschoben
    Parameter 4. und 5. sind fix
    """
    startpar = [1, 1, 1, s, l]
    ifixb = [1, 1, 1, 0, 0]  # fit should only vary m[0], m[1], m[2]
    
    #Data
    kr, sigsqr, delsigsqr = k[k > 0], sigsq[k > 0], delsigsq[k > 0]
    kl, sigsql, delsigsql  = np.abs(k[k < 0]), sigsq[k < 0], delsigsq[k < 0]
    
    # Default results, if fit fails
    epsr, depsr, betar = 0, 0, [0, 0, 0, s, l]
    epsl, depsl, betal = 0, 0, [0, 0, 0, s, l]
    fitted_sigmasqr = np.empty([len(kr)])
    fitted_sigmasql = np.empty([len(kl)])

    if k[np.where(sigsq == np.amin(sigsq))[0][0]] < 0:
        print("Minimum liegt im negativem Bereich __________________________________________________")
        if not kr.size == 0:
            epsr, depsr, betar = fitparabel(fitfkt2, kr, sigsqr, delsigsqr, [startpar, ifixb], "rechtem")
            fitted_sigmasqr = fitfkt2(betar, kr)
        if not kl.size == 0:
            epsl, depsl, betal = fitparabel(fitfkt1, kl, sigsql, delsigsql, [startpar, ifixb], "linken")
            fitted_sigmasql = fitfkt1(betal, kl)
            
    else:
        print("Minimum liegt im positivem Bereich __________________________________________________")
        if not kr.size == 0:
            epsr, depsr, betar = fitparabel(fitfkt1, kr, sigsqr, delsigsqr, [startpar, ifixb], "rechtem")
            fitted_sigmasqr = fitfkt1(betar, kr)
        if not kl.size == 0:
            epsl, depsl, betal = fitparabel(fitfkt2, kl, sigsql, delsigsql, [startpar, ifixb], "linken")
            fitted_sigmasql = fitfkt2(betal, kl)

    mean_eps = np.mean([epsl, epsr])
    dmean_eps = np.sqrt(depsl**2+depsr**2)/2

    result = {'epsr': epsr, 'depsr': depsr, 'betar': betar, 'fitted_sigmasqr': fitted_sigmasqr, 
        'epsl': epsl, 'depsl': depsl, 'betal': betal, 'fitted_sigmasql': fitted_sigmasql, 'mean_eps': mean_eps, 'dmean_eps': dmean_eps}
    return result