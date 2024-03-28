#Code :)
import numpy as np
from scipy.optimize import fsolve
import param

#KL Values
t_line_kl = .9
t_branch_kl = 2
ekl = .9
gv = 10 # open globe valve
sink_exit = 10
toliet_exit = 14
rho = 62.3 #lbm/ft3
mu = 6.733E0-4 #lbm/ft*s
g = 32.174 #ft/s^2
Vdot = 1 #gal/min
delz = 20 #ft

#Velocity Equation
def v(Vdot, D):
  return Vdot*np.pi/4 * D**2

#Reynolds Funciton
def Re(rho, v, D, mu):
  return rho * v * D / mu

#Colebrook Equaiton for friction
def f_cb(esp, D, Re, f, fguess):
    def resid(fg):
        return -2*np.log10((2.51/(Re*np.sqrt(fg)))) - (1/np.sqrt(fg))
    return fsolve(resid, fguess)[0]

#Head loss equation
#You'll need to set kL equal to an array for this to work
def Hp(delz, f, L, D, Kl, v, g):
  return delz + (f*(L/D) + np.sum(kl)v**2/(2*g)
