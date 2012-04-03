'''
@author: Daniel Machado

Copyright 2012

IBB-CEB:
Institute for Biotechnology and Bioengineering
Centre of Biological Engineering
University of Minho
 
This is free software: you can redistribute it and/or modify 
it under the terms of the GNU Public License as published by 
the Free Software Foundation, either version 3 of the License, or 
(at your option) any later version. 
 
This code is distributed in the hope that it will be useful, 
but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
GNU Public License for more details. 

You should have received a copy of the GNU Public License 
along with this code. If not, see http://www.gnu.org/licenses/ 
'''

import sys
from fractions import gcd
from time import time
from random import random
from numpy import array, nonzero, hstack, vstack, eye,sign, argsort, sum, abs, zeros
from numpy.linalg import matrix_rank
from scipy.io import savemat
from libsbml import SBMLReader #@UnresolvedImport


def main():
    """ Implements an elementary mode sampling method.
    For more information please consult the respective paper.
    Publication (Machado et al, 2012) submitted.
    If you use this in your work please cite the paper.
    
    Inputs: SBML model file, K (tunable parameter)
    Outputs: EM matrix as a .mat file.
    """
    
    
    if len(sys.argv) != 4:
        print "USAGE: python main.py [sbml model file] [K value] [output.mat file]"
    else:
        try:
            S, rev = load_sbml_model(sys.argv[1])
            K = int(sys.argv[2])
            E = sampler(S, rev, K, True)
            savemat(sys.argv[3], {'E' : E}, oned_as='row')
        except Exception as e:
            print e


def load_sbml_model(sbmlfile):
    """ Loads a metabolic model from an SBML file.
    Make sure that the model is balanced. If it contains external metabolites, remove before using.
    If you use constraint-based models, make sure the reversibility tag is consistent with the flux bounds.
    """

    reader = SBMLReader()
    document = reader.readSBML(sbmlfile)
    model = document.getModel()
    
    if model is None:
        raise IOError('Failed to load model.')

    metabolites = [species.getId() for species in model.getListOfSpecies()]

    S = [[0]*len(model.getListOfReactions()) for i in metabolites]

    for j, reaction in enumerate(model.getListOfReactions()):
        for reactant in reaction.getListOfReactants():
            i = metabolites.index(reactant.getSpecies())
            S[i][j] = -reactant.getStoichiometry()
        for product in reaction.getListOfProducts():
            i = metabolites.index(product.getSpecies())
            S[i][j] = product.getStoichiometry()
            
    rev = [reaction.getReversible() for reaction in model.getListOfReactions()]
    
    return S, rev


def sampler(S, rev, K=None, debug=False):
    """ Implements an elementary mode sampler.
    Inputs: S (stoichiometric matrix), rev (reversibility vector), K (tunable parameter to adjust output size).
    Note: if K is undefined it will compute all modes.
    For more information consult the respective paper (Machado et al; 2012) (submitted).
    """
    

    p = (lambda N: K / (K + float(N))) if K else (lambda N: 1)
    
    S = array(S)
    order = argsort(sum(abs(sign(S)),1))
    S = S[order,:]
    (m, n) = S.shape
    T = hstack((S.T, eye(n)))
    revT = array(rev)
    
    if debug:
        tstart = time()
        print 'starting computation...'
        
    for i in range(m):                
        keepT = nonzero(T[:,0] == 0)[0]
        delT = nonzero(T[:,0])[0]
        
        T2 = zeros((0, T.shape[1]-1))
        revT2 = zeros((0,))
       
        nrev = len(nonzero(revT[delT])[0])
        irr = nonzero(revT == 0)[0]
        npos = len(nonzero(T[irr,0] > 0)[0])
        nneg = len(nonzero(T[irr,0] < 0)[0])
        npairs = nrev*(nrev-1)/2 + nrev*(npos+nneg) + npos*nneg
        
        if debug:
            keep = keepT.shape[0]
            print 'line {} of {} - keep: {} combinations: {}'.format(i+1, m, keep, npairs),

        pairs = ((j, k) for j in delT for k in delT
            if k > j and (revT[j] or revT[k] or T[j,0]*T[k,0] < 0))
        
        Si = S[:i+1,:]

        for j, k in pairs:

            if revT[j] and revT[k]:
                cj, ck = -T[k,0], T[j,0]
            elif revT[j] and not revT[k]:
                cj, ck = -sign(T[j,0])*T[k,0], sign(T[j,0])*T[j,0]
            elif not revT[j] and revT[k]:
                cj, ck = sign(T[k,0])*T[k,0], -sign(T[k,0])*T[j,0]
            else:
                cj, ck = abs(T[k,0]), abs(T[j,0])

            Tj, Tk = cj * T[j,1:], ck * T[k,1:]
            Tjk = Tj + Tk 
            revTjk = revT[j] and revT[k]
            
            minimal = all(abs(Tj[(m-i-1):]) + abs(Tk[(m-i-1):]) == abs(Tjk[(m-i-1):])) \
                and _rank_test(Si, nonzero(Tjk[(m-i-1):])[0])
                       
            if minimal:
                T2 = vstack((T2,Tjk))
                revT2 = hstack((revT2,revTjk))
                                
        t = p(T2.shape[0]) if T2.shape[0] else 0
        selection = [i for i in range(T2.shape[0]) if random() <= t]
        T = vstack((T[keepT,1:],T2[selection,:])) if selection else T[keepT,1:]
        revT = hstack((revT[keepT],revT2[selection])) if selection else revT[keepT]

        if debug:
            total = T.shape[0]
            new = total - keep
            print 'new: {} total {}'.format(new, total)
            
    if debug:
        tend = time()
        print 'computation took:', (tend - tstart)
        print 'post-processing... ', 
                
    E = map(_normalize, T)
    E = map(lambda e: e.tolist(), E)
    
    if debug:
        print 'done' 
        print 'Found {} modes.'.format(len(E))
    
    return E


def _rank_test(S, Sjk):
    if len(Sjk) > S.shape[0] + 1:
        return False
    else:
        return len(Sjk) - matrix_rank(S[:,Sjk]) == 1


def _normalize(e):
    support = abs(e[nonzero(e)[0]])
    n1 = reduce(gcd, support) # greatest common denominator
    n2 = (min(support) * max(support)) ** 0.5 # geometric mean
    n = n1 if (1e-6 < n1 < 1e6) and (1e-6 < n2 < 1e6) else n2
    return e / n


if __name__ == '__main__':
    main()