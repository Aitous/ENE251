#!/usr/bin/env python
# coding: utf-8

# In[38]:


import time
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from scipy import interpolate

# Solving RachfordRice

def SolveRachfordRice(l, Nc, z, K):
    F  = lambda l: sum([(1 - K[i]) * z[i]/(K[i] + (1 - K[i]) * l) for i in range(Nc)])
    dF = lambda l: sum([-z[i] * (1 - K[i])**2/((K[i] + (1 - K[i]) * l)**2) for i in range(Nc)])

    F0 = F(0)
    F1 = F(1)
    
    if(F0 > 0 and F1 < 0):
        lmin = 0
        lmax = 1
    elif(F0 > 0 and F1 > 0):
        lmin = 1
        lmax = np.max([(K[i]*z[i] - K[i])/(1 - K[i]) for i in range(Nc)])
        
    else:
        lmax = 0
        lmin = np.min([(z[i] - K[i])/(1 - K[i]) for i in range(Nc)])

    useNewton = True                                #Change to false for bisection only
    error = []                                      #error array
    i = 0
    tol = 1.e-5

    while abs(F(l)) > tol:
        if(F(l) > 0):
            lmin = l
        else: 
            lmax = l
        delta_l = - F(l) / dF(l)
        if(l + delta_l > lmin and l + delta_l < lmax and useNewton):
            l = l + delta_l
        else:
            l = 0.5 * (lmin + lmax)
        error.append(F(l))
        #print('error = ', error[i])                 #reporting error for each step
        i += 1
        
    return l

#Calculating the a's and b's of the vapor and liquid phases. The function kij loads the interaction coefficients based on the EOS of interest 

def kij(EOS):
    if EOS is 'PR':
        return np.zeros((19,19))
    elif EOS is 'SRK':
        return np.array([[0 , 0.1, 0.1257, 0.0942],                         [0.1, 0, 0.027, 0.042],                         [0.1257, 0.027, 0, 0.008],                         [0.0942, 0.042, 0.008, 0]])
        return Kij            
    elif EOS is 'RK':
        return np.zeros([3,3])

def calc_a(EOS, T, Tc, Pc, omega):
    '''calculates ai for each component for the EOS of interest
       EOS: Equation of state (PR, SRK, or RK)
       T, Tc: temperature and critical temperature of the component
       Pc: critical pressure of the component
       omega: accentric factor for the component'''

    R = 8.314

    if EOS is 'PR':
        fw = 0.37464 + 1.54226*omega - 0.26992*omega**2
        a1 = np.divide(0.45724*R**2*Tc**2 , Pc)
        a2 = (1 + np.multiply(fw, (1 - np.sqrt(np.divide(T, Tc)))))**2
        a  = np.multiply(a1, a2)
    elif EOS is 'SRK':
        fw = 0.48 + 1.574*omega - 0.176*omega**2
        a1 = np.divide((0.42748*R**2*Tc**2), Pc)
        a2 = (1 + np.multiply(fw, (1 - np.sqrt(np.divide(T, Tc)))))**2
        a  = np.multiply(a1, a2)
    elif EOS is 'RK':
        a = np.divide(0.42748*R**2*Tc**(5/2), (Pc*T**0.5))
    else:
        print('parameters for his EOS is not defined')

    return a


def calc_b(EOS, Tc, Pc):
    '''calculates bi for each component for the EOS of interest
    EOS: Equation of state (PR, SRK, or RK)
    Tc: critical temperature of the component
    Pc: critical pressure of the component
    '''

    R = 8.314 # gas constant

    # The below if statement computes b for each 
    # componenet based on the EOS of
    # interest (Table 5.1 in the course reader)
    if EOS is 'PR':
        b = np.divide(0.07780*R*Tc, Pc)
    elif EOS is 'SRK':
        b = np.divide(0.08664*R*Tc, Pc)
    elif EOS is 'RK':
        b = np.divide(0.08664*R*Tc ,Pc)
    return b

def find_am(EOS, y, T, Tc, Pc, omega):
    ''' calculates the a parameter for the EOS of interest
        EOS: equation of state of interest (PR, SRK, RK)
        y: vapor or liquid compositions
        T, Tc: temperature value and critical temperature array
        Pc: critical pressure array
        omega: accentric factors array '''
    kijs = kij(EOS)
    am = np.sum(y[i]*y[j]*np.sqrt(calc_a(EOS, T, Tc[i], Pc[i], omega[i])               *calc_a(EOS, T, Tc[j], Pc[j], omega[j]))*(1-kijs[i,j])                for i in range(len(y)) for j in range(len(y)))
    return am

def find_bm(EOS, y, Tc, Pc):
    '''This function computes the b for the mixture for the EOS of interest
        EOS: Equation of state (PR, SRK, or RK)
        y: liquid or vapor compositions array
        Tc and Pc: critical temperature and pressure array
    '''
    bm = np.sum(np.multiply(y, calc_b(EOS, Tc, Pc)))
    return bm

def Z_factor(EOS, P, T, a, b):
    '''This function computes the Z factor for the cubic EOS of interest
       EOS: equation of state (PR, SRK, or RK)
       P, T: pressure and temperature
       a, b: the vapor or liquid parameters of equation of state
    '''

    R = 8.314 # gas constant

    if EOS == 'PR':
        u = 2
        w = -1
    elif EOS == 'SRK':
        u = 1
        w = 0
    elif EOS == 'RK':
        u = 1
        w = 0

    A = np.divide(a*P, R**2*T**2)
    B = np.divide(b*P, R*T)

    Coeffs = list()
    Coeffs.append(1)
    Coeffs.append(-(1 + B - u*B))
    Coeffs.append(A + w*B**2 - u*B - u*B**2)
    Coeffs.append(-np.multiply(A, B) - w*B**2 - w*B**3)

    Z = np.roots(Coeffs)
    # remove the roots with imaginary parts
    Z = np.real(Z[np.imag(Z) == 0])
    Zv = max(Z)
    Zl = min(Z)
    return Zv, Zl

def get_fug(EOS, y, Z, Tc, Pc, P, T, omega, a, b):
    '''This function computes the liquid or vapor fugacity of all components
       using Eq. 6.8 in course reader
       parameters needed:
       EOS: equation of state (PR, SRK, or RK)
       y: liquid or vapor compositions
       Z: z-factors for vapor or liquid
       Tc and Pc: critical temperature and pressure for all individual comp.s
       P, T: pressure and temperature of the system
       omega: accentric factors for all individual components
       a and b: EOS parameters as computed in another function
    '''
    R = 8.314 # gas constant

    if EOS is 'PR':
        u = 2
        w = -1
        kijs = kij(EOS)
    elif EOS is 'SRK':
        u = 1
        w = 0
        kijs = kij(EOS)
    elif EOS is 'RK':
        u = 1
        w = 0
        kijs = kij(EOS)
        
    fug = np.zeros(y.shape)
    A = np.divide(a*P, R**2*T**2)
    B = np.divide(b*P, R*T)
    delta_i = list()
    a_i = list()
    for i in range(len(y)):
        a_i.append(calc_a(EOS, T, Tc[i], Pc[i], omega[i]))
    for i in range(len(y)):  
        xa = 0
        for j in range(len(y)):
            xa += y[j] * math.sqrt(a_i[j]) * (1 - kijs[i][j])
        delta_i.append(2 * math.sqrt(a_i[i]) / a * xa)
    for i in range(len(fug)):
        bi = calc_b(EOS, Tc, Pc)[i]
        ln_Phi = bi/b * (Z - 1) - math.log(Z - B)                     + A / (B * math.sqrt(u**2 - 4*w)) * (bi/b - delta_i[i]) * math.log((2 * Z + B *(u + math.sqrt(u**2 - 4*w))) /(2 * Z + B *(u - math.sqrt(u**2 - 4*w))))
        fug[i] = y[i] * P * math.exp(ln_Phi)

    return fug

def Ki_guess(Pc, Tc, P, T, omega, Nc):
    Ki = np.array([Pc[i]/P * np.exp(5.37 * (1 + omega[i]) * (1 - Tc[i]/T)) for i in range(Nc)])
    return Ki

def flash(EOS, l, Nc, zi, Tc, Pc, P, T, omega):
    Ki = Ki_guess(Pc, Tc, P, T, omega, Nc) 
    tol = 1e-5
    R = 8.314 # gas constant
    l = SolveRachfordRice(l, Nc, zi, Ki)
            
    xi = np.divide(zi, l+(1-l)*Ki)
    yi = np.divide(np.multiply(Ki, zi), (l+(1-l)*Ki))
            
    av = find_am(EOS,yi,T,Tc,Pc,omega)
    al = find_am(EOS,xi,T,Tc,Pc,omega)
        
    bv = find_bm(EOS,yi,Tc,Pc)
    bl = find_bm(EOS,xi,Tc,Pc)
    
    #Z and fugacity determination for the vapor phase based on minimising Gibbs Free Energy      
    Zv = Z_factor(EOS,P,T,av,bv) #containing the max and min roots
    fugV_v = get_fug(EOS, yi, Zv[0], Tc, Pc, P, T, omega, av, bv)
    fugV_l = get_fug(EOS, yi, Zv[1], Tc, Pc, P, T, omega, av, bv)
    deltaGV = np.sum(yi * np.log(fugV_l / fugV_v))
    if deltaGV <= 0:
        Zv = Zv[1]
        fug_v = fugV_l
    else:
        Zv = Zv[0]
        fug_v = fugV_v

    #Z and fugacity determination for the liquid phase based on minimising Gibbs Free Energy   
    Zl = Z_factor(EOS,P,T,al,bl) #containing the max and min roots
    fugL_v = get_fug(EOS, xi, Zl[0], Tc, Pc, P, T, omega, al, bl)
    fugL_l = get_fug(EOS, xi, Zl[1], Tc, Pc, P, T, omega, al, bl)
    deltaGL = np.sum(xi * np.log(fugL_l / fugL_v))
    if deltaGL <= 0:
        Zl = Zl[1]
        fug_l = fugL_l
    else:
        Zl = Zl[0]
        fug_l = fugL_v
        
    while np.max(abs(np.divide(fug_v, fug_l) - 1)) > tol:
        Ki = Ki * np.divide(fug_l, fug_v)
        l = SolveRachfordRice(l, Nc, zi, Ki)
            
        xi = np.divide(zi, l+(1-l)*Ki)
        yi = np.divide(np.multiply(Ki, zi), (l+(1-l)*Ki))
                
        av = find_am(EOS,yi,T,Tc,Pc,omega)
        al = find_am(EOS,xi,T,Tc,Pc,omega)
            
        bv = find_bm(EOS,yi,Tc,Pc)
        bl = find_bm(EOS,xi,Tc,Pc)
        
        #Z and fugacity determination for the vapor phase based on minimising Gibbs Free Energy      
        Zv = Z_factor(EOS,P,T,av,bv) #containing the max and min roots
        fugV_v = get_fug(EOS, yi, Zv[0], Tc, Pc, P, T, omega, av, bv)
        fugV_l = get_fug(EOS, yi, Zv[1], Tc, Pc, P, T, omega, av, bv)
        deltaGV = np.sum(yi * np.log(fugV_l / fugV_v))
        if deltaGV <= 0:
            Zv = Zv[1]
            fug_v = fugV_l
        else:
            Zv = Zv[0]
            fug_v = fugV_v
    
        #Z and fugacity determination for the liquid phase based on minimising Gibbs Free Energy   
        Zl = Z_factor(EOS,P,T,al,bl) #containing the max and min roots
        fugL_v = get_fug(EOS, xi, Zl[0], Tc, Pc, P, T, omega, al, bl)
        fugL_l = get_fug(EOS, xi, Zl[1], Tc, Pc, P, T, omega, al, bl)
        deltaGL = np.sum(xi * np.log(fugL_l / fugL_v))
        if deltaGL <= 0:
            Zl = Zl[1]
            fug_l = fugL_l
        else:
            Zl = Zl[0]
            fug_l = fugL_v
            
        Vv = np.divide(Zv*R*T, P)
        Vl = np.divide(Zl*R*T, P)
    
    return (fug_v, fug_l, l, xi, yi)

def volumeCorrection(EOS, V, zi, Pc, Tc):
    Mw = np.array([44.01, 28.013, 16.043, 30.07, 44.097, 58.123, 58.123, 72.15, 72.15, 84, 96,                 107, 121, 134, 163.5, 205.4, 253.6, 326.7, 504.4])
    if EOS == "PR":
        #Si from the reader page 129
        S = [-0.1540, 0.1002, -0.08501, -0.07935, -0.06413, -0.04350, -0.04183, -0.01478]
        c = [3.7, 0] #CO2 and N2
        #For the heavy components
        for i in range(10, len(Pc)):
            S.append(1 - 2.258/Mw[i]**0.1823) #values correlated for heavier components (+C7)
        for i in range(0, len(Pc)-2):
            c.append(S[i] * calc_b(EOS, Tc[i+2], Pc[i+2]))
        V = V - np.sum([zi[i] * c[i] for i in range(len(Pc))])
        return V

def volume(EOS, P, T, Pc, Tc, omega, zi = np.array([1]), mixture = False):
    R = 8.314
    if not mixture:
        a = calc_a(EOS, T, Tc, Pc, omega)
        b = calc_b(EOS, Tc, Pc)
        Z = Z_factor(EOS,P,T,a,b)
        fug_v = get_fug(EOS, zi, Z[0], Tc, Pc, P, T, omega, a, b)
        fug_l = get_fug(EOS, zi, Z[1], Tc, Pc, P, T, omega, a, b)
        deltaG = np.sum(zi * np.log(fug_l / fug_v))
        if deltaG <= 0:
            Z = Z[1]
            fug = fug_l
        else:
            Z = Z[0]
            fug = fug_v
        V = np.divide(Z*R*T, P)
    else:
        bm = find_bm(EOS, zi, Tc, Pc)
        am = find_am(EOS, zi, T, Tc, Pc, omega)
        Z = Z_factor(EOS,P,T,am,bm)
        fug_v = get_fug(EOS, zi, Z[0], Tc, Pc, P, T, omega, am, bm)
        fug_l = get_fug(EOS, zi, Z[1], Tc, Pc, P, T, omega, am, bm)
        deltaG = np.sum(zi * np.log(fug_v / fug_l))
        if deltaG <= 0:
            Z = Z[1]
            fug = fug_l
        else:
            Z = Z[0]
            fug = fug_v
        V = np.divide(Z*R*T, P)
        #V = volumeCorrection(EOS, V, zi, Pc, Tc)
    
    return V

# Computes reference viscosity of methane using the correlation of Hanley et al. Cyrogenics, July 1975
# To be used for corresponding states computation of mixture viscosity
# A. R. Kovscek
# 20 November 2018

    
# Tref is the reference temperature in K (viscosity computed at this temperature)
# rho_ref is the reference density in g/cm3 (viscosity computed at this temperature and density)
# mu_C1 is the viscosity from correlation in mPa-s (identical to cP)
def ViscMethane(Tref,rho_ref):
    import math
    #Local variables
    #critical density of methane (g/cm^3)
    rho_c=16.043/99.2
    #parameters for the dilute gas coefficient
    GV=[-209097.5,264726.9,-147281.8,47167.40,-9491.872,1219.979,-96.27993,4.274152,-0.08141531]
    #parameters for the first density correction term
    Avisc1 = 1.696985927
    Bvisc1 = -0.133372346
    Cvisc1 = 1.4
    Fvisc1 = 168.0
    #parameters for the viscosity remainder
    j1 = -10.35060586
    j2 = 17.571599671
    j3 = -3019.3918656
    j4 = 188.73011594
    j5 = 0.042903609488
    j6 = 145.29023444
    j7 = 6127.6818706
    #compute dilute gas coefficient
    visc0 = 0.
    exp1 = 0.
    for i in range(0,len(GV)):
        exp1 = -1. + (i)*1./3.
        visc0 = visc0 + GV[i]*math.pow(Tref,exp1)
    #first density coefficient
    visc1 = Avisc1+Bvisc1*math.pow((Cvisc1-math.log(Tref/Fvisc1)),2.)
    #viscosity remainder
    theta=(rho_ref-rho_c)/rho_c
    visc2 = math.pow(rho_ref,0.1)
    visc2 = visc2*(j2+j3/math.pow(Tref,1.5))+theta*math.sqrt(rho_ref)*(j5+j6/Tref+j7/math.pow(Tref,2.))
    visc2 = math.exp(visc2)
    visc2 = visc2 - 1.
    visc2 = math.exp(j1+j4/Tref)*visc2
    #methane viscosity at T and density (Tref,rho_ref)
    #multiply by 10-4 to convert to mPa-s(cP)
    mu_C1 = (visc0+visc1+visc2)*0.0001
       
    return (mu_C1)

def get_interp_density(p, T):
    '''
    @param p: pressure in Pa
    @param T: temperature in K
    @return : methane density in kg/m3
    '''
    data_p = [0.1e6, 1e6, 3e6, 5e6, 10e6, 20e6, 50e6]
    data_T = [90.7, 94, 98, 100, 105, 110, 120, 140, 170]
    
    if p < data_p[0] or p > data_p[-1] or T < data_T[0] or T > data_T[-1]:
        raise Exception('Input parameter out of range')
    
    data_den = [[451.5, 447.11, 441.68, 438.94, 431.95, 424.79, 409.9, 1.403, 1.1467],
                [451.79, 447.73, 442.34, 439.62, 432.7, 425.61, 410.9, 377.7, 14.247],
                [453, 449.08, 443.78, 441.11, 434.32, 427.38, 413.05, 381.12, 314.99],
                [454, 450.4, 445.19, 442.55, 435.89, 429.09, 415.1, 384.28, 324.32],
                [456, 453.57, 448.55, 446.02, 439.63, 433.13, 419.9, 391.35, 340.6],
                [460, 458, 454.74, 452.37, 446.43, 440.43, 428.32, 402.99, 361.57],
                [477, 473, 470, 468, 463.2, 458.14, 448.08, 427.88, 397.48]]
    f = interpolate.interp2d(data_T, data_p, data_den)
    return f(T, p)


# In[39]:


def LBC_viscosity(P, T, zi, Tc, Pc, omega, Mw, Vci):
    coef = [0.10230, 0.023364, 0.058533, -0.040758, 0.0093324]
    
    Nc = len(zi)
    EOS = 'PR'
    Pmax = 3000 * 6894.76
    Pressure = []
    visc = []
    while P < Pmax:
        #flash
        fug_v, fug_l, l, xi, yi = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
        if l>1:
            xi = zi
        #Computing Ksi
        Ksi = 5.4402 * 399.54 * np.sum(xi * Tc)**(1/6)/np.multiply(np.sum(xi * Mw)**(0.5),np.sum(xi * Pc)**(2/3))
        Ksi_i = 5.4402 * 399.54 * Tc**(1/6) * Mw**(-0.5) * Pc**(-2/3)
        #Ksi_i = Tc**(1/6) * Mw**(-0.5) * Pc**(-2/3)
        eta_star_i = np.zeros(xi.shape)
        
        for i in range(Nc):
            Tr = T/Tc[i]
            if Tr < 1.5:
                eta_star_i[i] = 34e-5 * (Tr**0.94)/Ksi_i[i]
            else:
                eta_star_i[i] = 17.78 * 1e-5 * ((4.58*Tr - 1.67)**0.625)/ Ksi_i[i]
        
        eta_star = np.divide(np.sum(xi * eta_star_i * Mw**0.5), np.sum(xi * Mw**0.5))
        MC7_plus = np.sum(xi[i] * Mw[i] for i in range(10, Nc)) / np.sum(xi[i] for i in range(10, Nc))
        denC7_plus = 0.895
        Vc_plus = (21.573 + 0.015122*MC7_plus - 27.656*denC7_plus + 0.070615*denC7_plus*MC7_plus) * 6.2372*1e-5
        V_mixture = volume(EOS, P, T, Pc, Tc, omega, xi, True)
        xC7_plus = np.sum(xi[i] for i in range(10, Nc))
        Vc_mixture = np.sum(xi[i] * Vci[i] for i in range(10))*1e-6 + xC7_plus * Vc_plus
        rho_r = Vc_mixture/V_mixture
        viscosity = ((coef[0] + coef[1] * rho_r + coef[2] * rho_r**2 + coef[3] * rho_r**3 + coef[4] * rho_r**4)**4                           - 0.0001)/Ksi + eta_star
        visc.append(viscosity)
        Pressure.append(P)
        P = 1.1 * P
    
    plt.plot(Pressure, visc)
    plt.xlabel("Pressure (Pa)")
    plt.ylabel("Viscosity (cP)")
    plt.title("Viscosity vs pressure")
    plt.show()


# In[40]:


P = 1500 * 6894.76
T = 106 + 273.15
Names = {'CO2' 'N2' 'C1' 'C2' 'C3' 'iC4' 'n-C4' 'i-C5' 'n-C5' 'C6' 'C7' 'C8' 'C9' 'C10' 'PS1' 'PS2' 'PS3' 'PS4' 'PS5'}
   
zi = np.array([0.0044, 0.0017, 0.3463, 0.0263, 0.0335, 0.092, 0.0175, 0.0089, 0.0101, 0.0152, 0.05,                 0.0602, 0.0399, 0.0355, 0.1153, 0.0764, 0.0633, 0.0533, 0.0330])#oil composition
Tc = np.array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4, 469.6, 507.4, 548,                 575, 603, 626, 633.1803, 675.9365, 721.3435, 785.0532, 923.8101]) # in Kelvin
Pc = np.array([72.9, 33.6, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3, 29.3, 30.7, 28.4, 26,                  23.9, 21.6722, 19.0339, 16.9562, 14.9613, 12.6979])*101325 # in Pa
omega  = np.array([0.228, 0.04, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227, 0.251, 0.296, 0.28,                 0.312, 0.348, 0.385, 0.6254, 0.7964, 0.9805, 1.2222, 1.4000]) # accentric factors
Mw = np.array([44.01, 28.013, 16.043, 30.07, 44.097, 58.123, 58.123, 72.15, 72.15, 84, 96,                 107, 121, 134, 163.5, 205.4, 253.6, 326.7, 504.4])
Vci = np.array([91.9, 84, 99.2, 147, 200, 259, 255, 311, 311, 368]) # cm3/mol

LBC_viscosity(P, T, zi, Tc, Pc, omega, Mw, Vci)


# In[41]:


def corresponding_state_Visco(P, T ,zi, Tc, Pc, omega, Mw):
    R = 8.314 # gas constant
    
    Names = {'CO2' 'N2' 'C1' 'C2' 'C3' 'iC4' 'n-C4' 'i-C5' 'n-C5' 'C6' 'C7' 'C8' 'C9' 'C10' 'PS1' 'PS2' 'PS3' 'PS4' 'PS5'}

    Nc = len(zi)
    EOS = 'PR'
    tol = 1e-5
    Pmax = 3000 * 6894.76
    visc = []
    Pressure = []
    while P < Pmax:
        fug_v, fug_l, l, xi, yi = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
        if l>1:
            xi = zi
        #Initiliazing
        Tc_mix = 0
        Mmix = 0
        Mn = 0
        M = 0
        denominator = 0
        for i in range(Nc):
            Mn += xi[i] * Mw[i]
            M += xi[i] * Mw[i]**2
            for j in range(Nc):
                Tc_mix += xi[i]*xi[j]*(Tc[i] * Tc[j])**(0.5)*((Tc[i]/Pc[i])**(1/3) + (Tc[j]/Pc[j])**(1/3))**3
                denominator += xi[i]*xi[j]*((Tc[i]/Pc[i])**(1/3) + (Tc[j]/Pc[j])**(1/3))**3
        Tc_mix = Tc_mix / denominator
        Pc_mix = 8 * Tc_mix / denominator
        M /= Mn
        Mmix = 1.304 * 1e-4 * (M**2.303 - Mn**2.303) + Mn
        Tr = (T * Tc[2])/Tc_mix
        Pr = (P * Pc[2])/Pc_mix
        rho_c = 162.84  #kg/m3
        #volume correction
        S = -0.154
        b = calc_b(EOS, Tc[2], Pc[2])
        Vc = volume(EOS, Pr, Tr, np.array([Pc[2]]), np.array([Tc[2]]), np.array([omega[2]]))
        volume_cor = Vc - b * S
        rho_r = Mw[2] * 1e-3 / volume_cor / rho_c
        
        alpha_mix = 1 + 7.378 * 10**(-3) * rho_r ** 1.847 * Mmix**0.5173
        alpha_0 = 1 + 0.031*rho_r**1.847
        
        Tref = Tr * alpha_0 / alpha_mix
        Pref = Pr * alpha_0 / alpha_mix
        
        S = -0.085
        Vc_ref = volume(EOS, Pref, Tref, np.array([Pc[2]]), np.array([Tc[2]]), np.array([omega[2]]))
        volume_cor = Vc_ref - b * S
        rho_ref = Mw[2]/volume_cor/ 1e6
        visc_methane = ViscMethane(Tref, rho_ref)
        
        visc_mix = (Tc_mix/Tc[2])**(-1/6) * (Pc_mix/Pc[2])**(2/3) * (Mmix/Mw[2])**(1/2) * alpha_mix / alpha_0 * visc_methane
        visc.append(visc_mix)
        Pressure.append(P)
        #print(P, visc_mix)
        P = 1.1 * P
    
    plt.plot(Pressure, visc)
    plt.xlabel("Pressure (Pa)")
    plt.ylabel("Viscosity (cP)")
    plt.title("Viscosity vs compositions")
    plt.show()


# In[42]:


zi = np.array([0.0044, 0.0017, 0.3463, 0.0263, 0.0335, 0.092, 0.0175, 0.0089, 0.0101, 0.0152, 0.05,                  0.0602, 0.0399, 0.0355, 0.1153, 0.0764, 0.0633, 0.0533, 0.0330])#oil composition

Tc = np.array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4, 469.6, 507.4, 548,              575, 603, 626, 633.1803, 675.9365, 721.3435, 785.0532, 923.8101]) # in Kelvin

Pc = np.array([72.9, 33.6, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3, 29.3, 30.7, 28.4, 26,               23.9, 21.6722, 19.0339, 16.9562, 14.9613, 12.6979])*101325 # in Pa

omega  = np.array([0.228, 0.04, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227, 0.251, 0.296, 0.28,              0.312, 0.348, 0.385, 0.6254, 0.7964, 0.9805, 1.2222, 1.4000]) # accentric factors

Mw = np.array([44.01, 28.013, 16.043, 30.07, 44.097, 58.123, 58.123, 72.15, 72.15, 84, 96,              107, 121, 134, 163.5, 205.4, 253.6, 326.7, 504.4])

P = 1500 * 6894.76
T = 106 + 273.15
corresponding_state_Visco(P, T, zi, Tc, Pc, omega, Mw)


# In[43]:


def viscosity(Oilcomp, Injcomp, P, T, Pc, Tc, omega, Mw, Vci):
    coef = [0.10230, 0.023364, 0.058533, -0.040758, 0.0093324]
    EOS = 'PR'
    Nc = len(Oilcomp)
    alpha = 0.5
    l = 0.5
    tol = 1e-5
    zi = Oilcomp + alpha * (Injcomp - Oilcomp)
    fug_v, fug_l, l, xi, yi = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
    Ksi = 5.4402 * 399.54 * np.sum(xi * Tc)**(1/6)/np.multiply(np.sum(xi * Mw)**(0.5),np.sum(xi * Pc)**(2/3))
    Ksi_i = 5.4402 * 399.54 * Tc**(1/6) * Mw**(-0.5) * Pc**(-2/3)
    eta_star_i = np.zeros(xi.shape)
                
    for i in range(Nc):
        Tr = T/Tc[i]
        if Tr < 1.5:
            eta_star_i[i] = 34e-5 * (Tr**0.94)/Ksi_i[i]
        else:
            eta_star_i[i] = 17.78 * 1e-5 * ((4.58*Tr - 1.67)**0.625)/ Ksi_i[i]
                
    eta_star = np.divide(np.sum(xi * eta_star_i * Mw**0.5), np.sum(xi * Mw**0.5))
    MC7_plus = np.sum(xi[i] * Mw[i] for i in range(10, Nc)) / np.sum(xi[i] for i in range(10, Nc))
    denC7_plus = 0.895
    Vc_plus = (21.573 + 0.015122*MC7_plus - 27.656*denC7_plus + 0.070615*denC7_plus*MC7_plus) * 6.2372*1e-5
    V_mixture = volume(EOS, P, T, Pc, Tc, omega, xi, True)
    xC7_plus = np.sum(xi[i] for i in range(10, Nc))
    Vc_mixture = np.sum(xi[i] * Vci[i] for i in range(10))*1e-6 + xC7_plus * Vc_plus
    rho_r = Vc_mixture/V_mixture
    viscosity = ((coef[0] + coef[1] * rho_r + coef[2] * rho_r**2 + coef[3] * rho_r**3 + coef[4] * rho_r**4)**4                    - 0.0001)/Ksi + eta_star
        
    return viscosity


# In[44]:


def vicosity_vs_composition(LPG_CO2_comb, Oilcomp, P, T, Pc, Tc, omega, Mw, Vci, makePlot = False):
    '''This function computes the MMP for the different compositions of the LPG and CO2
        and returns a plot of the MMP versus gas injectant composition.
        LPG_CO2_comb contains the porcentage of LPG in the mixte.
        an array of the form [0.7, 0.55, 0.4, 0.2, 0.1, 0.] means that for the first mixture
        we have 70% LPG and 30% CO2 and for the second 55% LPG and 45% CO2 and so on...
        The LPG composition is: C2: 0.01, C3: 0.38, iC4: 0.19, nC4: 0.42.
    '''
    #reservoir Oil components.
    Names = {'CO2' 'N2' 'C1' 'C2' 'C3' 'i-C4' 'n-C4' 'i-C5' 'n-C5' 'C6' 'C7' 'C8' 'C9' 'C10' 'PS1' 'PS2' 'PS3' 'PS4' 'PS5'}
    
    LPG = np.array([0, 0, 0, 0.01, 0.38, 0.19, 0.42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    CO2 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    numMixtures = len(LPG_CO2_comb)
    Viscosity = []
    composition = []
    
    for i in range(numMixtures):
        Injcomp = np.array(LPG_CO2_comb[i] * LPG + (1 - LPG_CO2_comb[i]) * CO2)
        Viscosity.append(viscosity(Oilcomp, Injcomp, P, T, Pc, Tc, omega, Mw, Vci))
        composition.append(LPG_CO2_comb[i])
        
    if makePlot:
        plt.plot(composition, Viscosity)
        plt.xlabel('Composition (mole fraction of the LPG)')  
        plt.ylabel('Viscosity (cP)')
        plt.title('Viscosty vs Injectant composition')
        plt.show()


# In[45]:


P = 28e6
T = 106 + 273.15
Names = {'CO2' 'N2' 'C1' 'C2' 'C3' 'iC4' 'n-C4' 'i-C5' 'n-C5' 'C6' 'C7' 'C8' 'C9' 'C10' 'PS1' 'PS2' 'PS3' 'PS4' 'PS5'}
   
zi = np.array([0.0044, 0.0017, 0.3463, 0.0263, 0.0335, 0.092, 0.0175, 0.0089, 0.0101, 0.0152, 0.05,                 0.0602, 0.0399, 0.0355, 0.1153, 0.0764, 0.0633, 0.0533, 0.0330])#oil composition
Tc = np.array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4, 469.6, 507.4, 548,                 575, 603, 626, 633.1803, 675.9365, 721.3435, 785.0532, 923.8101]) # in Kelvin
Pc = np.array([72.9, 33.6, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3, 29.3, 30.7, 28.4, 26,                  23.9, 21.6722, 19.0339, 16.9562, 14.9613, 12.6979])*101325 # in Pa
omega  = np.array([0.228, 0.04, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227, 0.251, 0.296, 0.28,                 0.312, 0.348, 0.385, 0.6254, 0.7964, 0.9805, 1.2222, 1.4000]) # accentric factors
Mw = np.array([44.01, 28.013, 16.043, 30.07, 44.097, 58.123, 58.123, 72.15, 72.15, 84, 96,                 107, 121, 134, 163.5, 205.4, 253.6, 326.7, 504.4])
Vci = np.array([91.9, 84, 99.2, 147, 200, 259, 255, 311, 311, 368]) # cm3/mol

vicosity_vs_composition(np.array([0.7, 0.55, 0.4, 0.2, 0.1, 0.]), zi, P, T, Pc, Tc, omega, Mw, Vci, True)


# In[47]:


#Plotting the experimental CCE curve of pressure against relative volume.
x = np.array([
20096156.97
,19406680.97
,18717204.97
,16834935.49
,15476667.77
,15312158.8
,13890872.97
,10443492.97

])
y = np.array([
0.622
,0.6162
,0.611
,0.597
,0.5869
,0.5857
,0.6341
,0.708
])
plt.plot(x, y, 'r')
plt.xlabel("Pressure (Pa)")
plt.ylabel("Viscosity (cP)")
plt.title("Experimental evolution of viscosity")
plt.legend(loc='best')
plt.show()


# In[ ]:




