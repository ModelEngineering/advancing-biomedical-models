# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 19:36:36 2021

@author: Herbert Sauro
"""

import tellurium as te
import roadrunner

r = te.loada("""
      J1: $Xo -> S1; k1*Xo;
      J2:  S1 -> S2; k2*S1
      J3: S2 -> S3; k3*S2;
      J4: S3 -> ;  k4*S3;
       
      J5: S1 -> S4; k5*S1;
      J6: S4 ->; k6*S4
      J7: S4 ->; k7*S4
       
       k1 = 0.1; k2 = 0.34; k3 = 0.23
       k4 = 0.9; k5 = 0.67; k6 = 0.55
       k7 = 0.1
       Xo = 10   
       
       at J5 > 0.65: k5 = 0.1
       at time > 8: k1 = 0
       at time > 20: k1 = 0.1
       at S2 > 0.7: k3 = 0.01
""")

m = r.simulate (0, 40, 100, ['time'] + r.getReactionIds())
r.plot()
r.reset()
m = r.simulate (0, 40, 100, ['time'] + r.getFloatingSpeciesIds())
r.plot()
