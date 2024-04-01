#Code :)

# Date 3/28/24 (James Burk) 

import numpy as np 
from scipy.optimize import fsolve

#marked out are to get it in SI units
rho = 62.3 #lbm/ft3
#rho = 997.9503 #kg/m^3
mu = 6.733E-4 #lbm/ft*s
#mu = 0.00100198075387 #Pa*s
g = 32.174 #ft/s^2
#g = 9.81 #m/s^2
Vdot = 1/60/7.48 #ft^3/sec
#Vdot = 1/15850 #m^3/sec
delz = 24 #ft
#delz = 20/3.28
kL = [.9, 2.0, 0.9, 10, 14, 10]
L = 27  #ft
#L = 23/3.28 #m
D = 1/12 # ft
#D = 1/12/3.28 #m

#Velocity Equation #ft/sec
def v(Vdot, D):
  return Vdot/(np.pi/4 * D**2)
vel = v(Vdot, D)
print('velocity =', vel)

#Reynolds Funciton
def Re(rho, v, D, mu):
  return rho * v * D / mu
Rey = Re(rho, vel, D, mu)
print('Reynolds Number=', Rey)

#Colebrook Equaiton for friction
def f_cb(D, Re, fguess):
    def resid(fg):
        return -2.0*np.log10((2.51/(Re*np.sqrt(fg)))) - (1/np.sqrt(fg))
    return fsolve(resid, fguess)[0]

f = f_cb(D, Rey, .0001)
print('friction =', f)

#Head loss equation
#You'll need to set kL equal to an array for this to work
def Hp():
  head = delz + (f*(L/D) + np.sum(kL))*vel**2/(2*g)
  return head

print('pump head=', Hp())


#Date 4/1/12 (Grace/James/Brian)
import numpy as np
from scipy.optimize import fsolve

D = 1/12 
Aa = (np.pi/4)*D**2
Ab = (np.pi/4)*D**2
Ac = (np.pi/4)*D**2
Ad = (np.pi/4)*D**2
Ae = (np.pi/4)*D**2
Af = (np.pi/4)*D**2

def equation(V):

  Vb, Vd, Vf = V

  vb = Vb/Ab
  vd = Vd/Ad
  vf = 1 ## put this into the right units and get this fsolve to work :)

  va = ((vb * Ab) + (vc * Ac)) / Aa
  vb = ((va * Aa) - (vc * Ac)) / Ab
  vc = ((va * Aa) - (vb * Ab)) / Ac
  vd = ((vc * Ac) - (ve * Ae)) / Ad
  ve = ((vc * Ac) - (vd * Ad)) / Ae
  vf = ((ve * Ae)) / Af

  velocities = [va, vb, vc, vd, ve, vf]

  return velocities

Vguess = [1,1,1]

answer = fsolve(equation,Vguess)

answer




## Bailey's Nonsense 4/1/24
#import necessary packages
import numpy as np
import scipy.optimize as opt

#Given Constants
rho = 62.3 #lbm/ft^3 - density of water
mu = 6.733e-4 #lbm/ft/s - viscosity of water
eps = 0 # Roughness factor
el = 0.9
tee_branch = 2
tee_line = 0.9

#Define Colebrook Function
def Colebrook(Re, eps, D, fguess):
  def resid(fg):
    leftside = 1 / np.sqrt(fg)
    rightside = -2.0 * np.log10(eps/D/3.7 + 2/51/Re/np.sqrt(fg))
    return leftside - rightside
  return opt.fsolve(resid,fguess)[0]

import numpy as np

def resid(guesses, KL, D, L, eps, rho, mu):
    num_pipes = len(KL)
    Vdot = guesses[:num_pipes]
    Vdot_tot = sum(Vdot)
    v = [Vdot[i] / (np.pi/4 * D[i]**2) for i in range(num_pipes)]
    Re = [rho * v[i] * D[i] / mu for i in range(num_pipes)]
    f = [Colebrook(Re[i], eps, D[i], 0.001) for i in range(num_pipes)]
    PL = [(f[i] * (L[i]/D[i]) + KL[i]) * v[i]**2 * rho / 2 for i in range(num_pipes)]
    resid1 = [PL[i+1] - PL[i] for i in range(num_pipes-1)]
    resid2 = [-PL[0] + P1]
    return resid1 + resid2
  
D = [0.75/12, 0.75/12, 0.75/12, 0.75/12, 0.75/12, 0.75/12, 0.75/12, 0.75/12, 0.75/12, 0.75/12, 0.75/12, 0.75/12, 0.75/12]
L = [20, 3, 4, 3, 4, 3, 4, 7.5, 4, 3, 4, 3, 4]
KL = [el, 0, tee_branch+el, tee_line, tee_branch+el, tee_line, tee_branch+el, 2*el, tee_branch+el, tee_line, tee_branch+el, tee_line, 2*el]

array_length = 13
value = 5
guesses = np.full(array_length, value)


# Reference pressure
P1 = 0

# Call the resid function
residuals = resid(guesses, KL, D, L, eps, rho, mu)
Vdot = opt.fsolve(resid,guesses,args = (KL, D, L, eps, rho, mu))
print("Residuals:", residuals)

Vdot
