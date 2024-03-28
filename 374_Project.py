#Code :)
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
