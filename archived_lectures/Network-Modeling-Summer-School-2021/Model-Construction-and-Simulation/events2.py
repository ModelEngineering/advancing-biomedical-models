# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 19:47:26 2021

@author: Herbert Sauro
"""

import tellurium as te
import roadrunner

r = te.loada("""
      $Xo -> S1; k1*Xo - k11*S1 
      S1 -> S2; k2*S1
      S2 -> S3; k3*S2
      S3 -> S4; k4*S3
      S4 -> S5; k4*S4
      S5 ->;  k5*S5
      
      k1 = 0.1; k11 = 0.05
      k2 = 0; k3 = 0; k4 = 0; k5 = 0
      Xo = 10
      
      at time >  100: k2 = 0.3
      at time >  130: k3 = 0.45
      at time >  160: k4 = 0.23
      at time >  180: k5 = 0.67

""")

m = r.simulate (0, 250, 100)
r.plot()
