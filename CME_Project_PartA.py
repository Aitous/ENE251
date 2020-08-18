#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve # you may or may not use this based on how you solve the Rachford Rice Equation


# # Solving ReachfordRice

# In[37]:


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


# Calculating the a's and b's of the vapor and liquid phases. The function kij loads the interaction coefficients based on the EOS of interest 

# In[38]:


def kij(EOS):
    if EOS[:-1] == 'PR':    #I added another char at the end of EOS. PRT, T for true means the use of Binary Coeff
                            # and PRF means using only zero binary coefficients.
        Kij = np.zeros((19, 19))
        if EOS[-1] == 'T':
            Kij[0] = [0, 0, 0.12, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
            Kij[1] = [0, 0, 0.02, 0.06, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]
            Kij[:, 0] = [0, 0, 0.12, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
            Kij[:, 1] = [0, 0, 0.02, 0.06, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]
            return Kij
        else: 
            return Kij
    elif EOS == 'SRK':
        Kij = np.zeros((19, 19))
        return Kij            
    elif EOS == 'RK':
        return np.zeros([19,19])


# In[39]:


def calc_a(EOS, T, Tc, Pc, omega):
    '''calculates ai for each component for the EOS of interest
       EOS: Equation of state (PR, SRK, or RK)
       T, Tc: temperature and critical temperature of the component
       Pc: critical pressure of the component
       omega: accentric factor for the component'''

    R = 8.314

    if EOS[:-1] == 'PR':
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
    if EOS[:-1] == 'PR':
        b = np.divide(0.07780*R*Tc, Pc)
    elif EOS is 'SRK':
        b = np.divide(0.08664*R*Tc, Pc)
    elif EOS is 'RK':
        b = np.divide(0.08664*R*Tc ,Pc)
    return b


# In[40]:


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


# In[41]:


def Z_factor(EOS, P, T, a, b):
    '''This function computes the Z factor for the cubic EOS of interest
       EOS: equation of state (PR, SRK, or RK)
       P, T: pressure and temperature
       a, b: the vapor or liquid parameters of equation of state
    '''

    R = 8.314 # gas constant

    if EOS[:-1] == 'PR':
        u = 2
        w = -1
    elif EOS is 'SRK':
        u = 1
        w = 0
    elif EOS is 'RK':
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


# In[42]:


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

    if EOS[:-1] == 'PR':
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
    #Z = Z[0]
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


# In[43]:


def Ki_guess(Pc, Tc, P, T, omega, Nc):
    Ki = np.array([Pc[i]/P * np.exp(5.37 * (1 + omega[i]) * (1 - Tc[i]/T)) for i in range(Nc)])
    return Ki


# In[47]:


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
        
    Vv = np.divide(Zv*R*T, P)
    Vl = np.divide(Zl*R*T, P)
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
        V = l*Vl + (1-l)*Vv
    if l > 1:
        V = Vl
    if l < 0:
        V = Vv
    return (fug_v, fug_l, l, V)


# In[48]:


def CME_SRK():
    #CME test using a SRK EOS and without binary coefficients. The EOS input parameters are found using 
    #an expanded plus fraction starting from C7.
    # The inputs of the problem:
    T = 106 + 273.15 # in Kelvin
    Pmin = 1e6      
    Pmax = 2e7
    P = Pmax
    R = 8.314 # gas constant

    Names = {'N2' 'CO2' 'C1' 'C2' 'C3' 'i-C4' 'n-C4' 'i-C5' 'n-C5' 'C6' 'PS1' 'PS2' 'PS3' 'PS4' 'PS5'}
    
    zi = np.array([0.0017, 0.0044, 0.3463, 0.0263, 0.0335, 0.092, 0.0175, 0.0089, 0.0101, 0.0152,                   0.2123, 0.1268, 0.0934, 0.0578, 0.0367]) # overall composition
    Tc = np.array([126.2, 304.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4, 469.6, 507.4,                   580.4641, 666.8868, 759.3859, 875.1269, 1114.0748]) # in Kelvin
    Pc = np.array([33.6, 72.9, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3, 29.3,                   25.5774, 20.5897, 16.7787, 14.2046, 11.9834])*101325 # in Pa
    omega  = np.array([0.04, 0.228, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227, 0.251, 0.296,                   0.4408, 0.6669, 0.8756, 1.0465, 0.8191]) # accentric factors
    Nc = zi.size # number of components
    
    EOS = 'SRK' # Equation of State we are interested to use
    tol  = 1e-5
    l = 0.5
    Volume = []
    Pressure = []
    #finding Pmax that we are going to use for bisection
    while(P > Pmin):
        fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
        if l > 1:
            Pmax = P  #saving the last pressure for which l is > 1
        Pressure.append(P)    
        Volume.append(V)
        P = 0.95 * P           #The multiplication coeff can be ajjusted depending on the desired precision.
        print("Liquid fraction: ",l)
    
    #Refining our guess of Psat by doing a bisection between Pmin and Pmax 
    while abs(l-1) > tol:
        P = (Pmin + Pmax) * 0.5
        fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
        if l > 1:
            Pmax = P
        else:
            Pmin = P
            
    lsat = l
    Psat = P
    Vsat = V
        
    print("liquid fraction is: {:.4f} and the bubble point pressure = {:.4f}".format(lsat, Psat))
    Volume = np.divide(Volume, Vsat)
    
    #Plotting the Pressure vs Volume curve
    plt.plot(Volume, Pressure, 'b-', label="P(V)")
    plt.xlim(0.9,1.4)
    plt.ylim(0.6e7, 2e7)
    plt.xlabel("Relative Volume")
    plt.ylabel("Pressure (Pa)")
    plt.title("Pressure against volume")
    plt.legend()


# In[49]:


CME_SRK()


# In[50]:


def CME_PR_First_Limping(UseBinaryCoeff):
    #This function treats the first lumping with an expansion starting from C7
    #An additional curve will be plotted for the case were BinaryCoeffs are not zero if UseBinaryCoeff = True
    #The inputs of the problem:
    T = 106 + 273.15 # in Kelvin
    Pmin = 1e6      
    Pmax = 2e7
    P = Pmax
    R = 8.314 # gas constant

    Names = {'N2' 'CO2' 'C1' 'C2' 'C3' 'i-C4' 'n-C4' 'i-C5' 'n-C5' 'C6' 'PS1' 'PS2' 'PS3' 'PS4' 'PS5'}
    
    zi = np.array([0.0017, 0.0044, 0.3463, 0.0263, 0.0335, 0.092, 0.0175, 0.0089, 0.0101, 0.0152,                   0.2123, 0.1268, 0.0934, 0.0578, 0.0367]) # overall composition
    Tc = np.array([126.2, 304.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4, 469.6, 507.4,                   580.4641, 666.8868, 759.3859, 875.1269, 1114.0748]) # in Kelvin
    Pc = np.array([33.6, 72.9, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3, 29.3,                   25.5774, 20.5897, 16.7787, 14.2046, 11.9834])*101325 # in Pa
    omega  = np.array([0.04, 0.228, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227, 0.251, 0.296,                   0.4408, 0.6669, 0.8756, 1.0465, 0.8191]) # accentric factors
    Nc = zi.size # number of components
    
    EOS = 'PRF' # Equation of State we are interested to use
    tol  = 1e-5
    l = 0.5
    Volume = []
    Pressure = []
    #finding Pmax that we are going to use for bisection
    while(P > Pmin):
        fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
        if l > 1:
            Pmax = P
        Pressure.append(P)    
        Volume.append(V)
        P = 0.95 * P
        print(l)
    
    #Refining our guess of Psat by doing a bisection between Pmin and Pmax 
    while abs(l-1) > tol:
        P = (Pmin + Pmax) * 0.5
        fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
        if l > 1:
            Pmax = P
        else:
            Pmin = P
            
    lsat = l
    Psat = P
    Vsat = V
    ######guessing Ki each time we change the pressure????######
    
        
    print("liquid fraction is: {:.4f} and the bubble point pressure = {:.4f}".format(lsat, Psat))
    Volume = np.divide(Volume, Vsat)
    print(Volume)
    print(Pressure)
    
    #Plotting the Pressure vs Volume curve
    plt.plot(Volume, Pressure, 'b-', label="P(V) without tunning")
    plt.xlim(0.9,1.4)
    plt.ylim(0.6e7, 2e7)
    plt.xlabel("Relative Volume")
    plt.ylabel("Pressure (Pa)")
    plt.title("Pressure against volume")
    plt.legend()
    
    if UseBinaryCoeff: 
        EOS = 'PRT' # Equation of State we are interested to use
        tol  = 1e-5
        l = 0.5
        Pmin = 1e6      
        Pmax = 2e7
        P = Pmax
        Volume = []
        Pressure = []
        #finding Pmax that we are going to use for bisection
        while(P > Pmin):
            fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
            if l > 1:
                Pmax = P
            Pressure.append(P)    
            Volume.append(V)
            P = 0.95 * P
            print(l)
        
        #Refining our guess of Psat by doing a bisection between Pmin and Pmax 
        while abs(l-1) > tol:
            P = (Pmin + Pmax) * 0.5
            fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
            if l > 1:
                Pmax = P
            else:
                Pmin = P
        ######guessing Ki each time we change the pressure????######
        lsat = l
        Psat = P
        Vsat = V
        
            
        print("liquid fraction is: {:.4f} and the bubble point pressure = {:.4f}".format(lsat, Psat))
        Volume = np.divide(Volume, Vsat)
        print(Volume)
        print(Pressure)
        
        #Plotting the Pressure vs Volume curve
        plt.plot(Volume, Pressure, 'r-', label="P(V) with tuning")
    plt.legend()
    plt.show()


# In[51]:


CME_PR_First_Limping(True)


# In[52]:


def CME_PR(usingHeptanePlusFraction):    
    #CME test using a PR EOS and without binary coefficients. The EOS input parameters are found using 
    #an expanded plus fraction starting from C10.
    #If usingHeptanePlusFraction is set to be true, the function plots another CME curve showing the case were 
    #the additional informations about the Heptane Plus fraction (we have its density which instead of C9 density
    #used in determining A, B, C and D values for the expansion).
    # The inputs of the problem:
    T = 106 + 273.15 # in Kelvin
    Pmin = 1e6 
    Pmax = 2e7
    P = Pmax
    R = 8.314 # gas constant

    Names = {'CO2' 'N2' 'C1' 'C2' 'C3' 'i-C4' 'n-C4' 'i-C5' 'n-C5' 'C6' 'C7' 'C8' 'C9' 'C10' 'PS1' 'PS2' 'PS3' 'PS4' 'PS5'}
    
    zi = np.array([0.0044, 0.0017, 0.3463, 0.0263, 0.0335, 0.092, 0.0175, 0.0089, 0.0101, 0.0152, 0.05,                  0.0602, 0.0399, 0.0355, 0.1153, 0.0764, 0.0633, 0.0533, 0.0330]) # overall composition
    Tc = np.array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4, 469.6, 507.4, 548,                  575, 603, 626, 633.1803, 675.9365, 721.3435, 785.0532, 923.8101]) # in Kelvin
    Pc = np.array([72.9, 33.6, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3, 29.3, 30.7, 28.4, 26,                   23.9, 21.6722, 19.0339, 16.9562, 14.9613, 12.6979])*101325 # in Pa
    omega  = np.array([0.228, 0.04, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227, 0.251, 0.296, 0.28,                  0.312, 0.348, 0.385, 0.6254, 0.7964, 0.9805, 1.2222, 1.4000]) # accentric factors
    Nc = zi.size # number of components
    
    #plotting the first part that uses a zero binary coefficient
    EOS = 'PRF' # Equation of State we are interested to use, 'T' at the end refer to not using binary coeffs
                #setting them to zero, see Kij(EOS)function
    tol  = 1e-5
    l = 0.5
    Volume = []
    Pressure = []
    #finding Pmax that we are going to use for bisection
    while(P > Pmin):
        fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
        if l > 1:
            Pmax = P           #saving the last pressure for which l > 1 to use it in bisection
        Pressure.append(P)    
        Volume.append(V)
        P = 0.97 * P           #The multiplication coeff can be ajjusted depending on the desired precision.
        print(l)
    
    #Refining our guess of Psat by doing a bisection between Pmin and Pmax 
    while abs(l-1) > tol:
        P = (Pmin + Pmax) * 0.5
        fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
        if l > 1:
            Pmax = P
        else:
            Pmin = P
    #reporting the values of lsat, Psat and Vsat        
    print("liquid fraction is: {:.4f} and the bubble point pressure = {:.4f}".format(l, P, V))
    Volume = np.divide(Volume, V)
    
    #Plotting the Pressure vs Volume curve
    plt.plot(Volume, Pressure, 'b-', label="P(V)")
    plt.xlim(0.9,1.4)
    plt.ylim(0.6e7, 2e7)
    plt.xlabel("Relative Volume")
    plt.ylabel("Pressure (Pa)")
    plt.title("Pressure against volume")
    plt.legend()
    
    #if th user chooses to compare with a computation that uses binary additional information about Heptane plus
    #fraction this should be set equal to true.
    if usingHeptanePlusFraction:   
        zi = np.array([0.0044, 0.0017, 0.3463, 0.0263, 0.0335, 0.092, 0.0175, 0.0089, 0.0101, 0.0152, 0.05,                  0.0602, 0.0399, 0.0355, 0.1153, 0.0764, 0.0633, 0.0533, 0.0330]) # overall composition
        Tc = np.array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4, 469.6, 507.4, 548,                  575, 603, 626, 635.6461, 677.1280, 721.3569, 783.6522, 920.0717]) # in Kelvin
        Pc = np.array([72.9, 33.6, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3, 29.3, 30.7, 28.4, 26,                   23.9, 21.6464, 19.0231, 16.9560, 14.9705, 12.7179])*101325 # in Pa
        omega  = np.array([0.228, 0.04, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227, 0.251, 0.296, 0.28,                  0.312, 0.348, 0.385, 0.6256, 0.7965, 0.9805, 1.2220, 1.3996]) # accentric factors
        Nc = zi.size # number of components
        
        
        EOS = 'PRF' # Equation of State we are interested to use, 'F' at the end refer to the use of binary coeffs
                    #that are non-zero, see Kij(EOS)function  
        Pmax = 2e7
        P = Pmax
        Pmin = 1e6 
        l = 0.5    
        Volume = []
        Pressure = []
        
        #finding Pmax that we are going to use for bisection
        while(P > Pmin):
            fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
            if l > 1:
                Pmax = P
            Pressure.append(P)    
            Volume.append(V)
            P = 0.95 * P
            print(l)
        
        #Refining our guess of Psat by doing a bisection between Pmin and Pmax 
        while abs(l-1) > tol:
            P = (Pmin + Pmax) * 0.5
            fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
            if l > 1:
                Pmax = P
            else:
                Pmin = P        
        #reporting the values of lsat, Psat and Vsat 
        print("liquid fraction is: {:.4f} and the bubble point pressure = {:.4f}".format(l, P, V))
        Volume = np.divide(Volume, V)
        print(Volume)
        print(Pressure)
        
        #Plotting the Pressure vs Volume curve
        plt.plot(Volume, Pressure, 'r-', label="P(V) with additional density info")
    plt.legend()
    plt.show()    
    


# In[53]:


CME_PR(False)


# In[54]:


def CME_PR_comparison_BinaryCoeffs(UseBinaryCoeff):
    #If UseBinaryCoeff is set to True, do routine compute 2 curves of pressure versus pressure. One for
    #the case without binary coefficients and the one for the case with those.
    # The inputs of the problem:
    T = 106 + 273.15 # in Kelvin
    Pmin = 1e6 
    Pmax = 2e7
    P = Pmax
    R = 8.314 # gas constant

    Names = {'CO2' 'N2' 'C1' 'C2' 'C3' 'i-C4' 'n-C4' 'i-C5' 'n-C5' 'C6' 'C7' 'C8' 'C9' 'C10' 'PS1' 'PS2' 'PS3' 'PS4' 'PS5'}
    
    zi = np.array([0.0044, 0.0017, 0.3463, 0.0263, 0.0335, 0.092, 0.0175, 0.0089, 0.0101, 0.0152, 0.05,                  0.0602, 0.0399, 0.0355, 0.1153, 0.0764, 0.0633, 0.0533, 0.0330]) # overall composition
    Tc = np.array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4, 469.6, 507.4, 548,                  575, 603, 626, 633.1803, 675.9365, 721.3435, 785.0532, 923.8101]) # in Kelvin
    Pc = np.array([72.9, 33.6, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3, 29.3, 30.7, 28.4, 26,                   23.9, 21.6722, 19.0339, 16.9562, 14.9613, 12.6979])*101325 # in Pa
    omega  = np.array([0.228, 0.04, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227, 0.251, 0.296, 0.28,                  0.312, 0.348, 0.385, 0.6254, 0.7964, 0.9805, 1.2222, 1.4000]) # accentric factors
    Nc = zi.size # number of components
    
    #plotting the first part that uses a zero binary coefficient
    EOS = 'PRF' # Equation of State we are interested to use, 'T' at the end refer to not using binary coeffs
                #setting them to zero, see Kij(EOS)function
    tol  = 1e-5
    l = 0.5
    Volume = []
    Pressure = []
    #finding Pmax that we are going to use for bisection
    while(P > Pmin):
        fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
        if l > 1:
            Pmax = P
        Pressure.append(P)    
        Volume.append(V)
        P = 0.99 * P
        print(l)
    
    #Refining our guess of Psat by doing a bisection between Pmin and Pmax 
    while abs(l-1) > tol:
        P = (Pmin + Pmax) * 0.5
        fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
        if l > 1:
            Pmax = P
        else:
            Pmin = P
    #reporting the values of lsat, Psat and Vsat        
    print("liquid fraction is: {:.4f} and the bubble point pressure = {:.4f}".format(l, P, V))
    Volume = np.divide(Volume, V)
    print(Volume)
    print(Pressure)
    
    #Plotting the Pressure vs Volume curve
    plt.plot(Volume, Pressure, 'b-', label="P(V) without tunning")
    plt.xlim(0.9,1.4)
    plt.ylim(0.6e7, 2e7)
    plt.xlabel("Relative Volume")
    plt.ylabel("Pressure (Pa)")
    plt.title("Pressure against volume")
    plt.legend()

    if UseBinaryCoeff:    #if th user chooses to compare with a computation that uses binary coefficients,
                          #should be set equal to true.
        #plotting the second part that uses a non-zero binary coefficient
        EOS = 'PRT' # Equation of State we are interested to use, 'T' at the end refer to the use of binary coeffs
                    #that are non-zero, see Kij(EOS)function  
        Pmax = 2e7
        P = Pmax
        Pmin = 1e6 
        l = 0.5    
        Volume = []
        Pressure = []
        
        #finding Pmax that we are going to use for bisection
        while(P > Pmin):
            fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
            if l > 1:
                Pmax = P
            Pressure.append(P)    
            Volume.append(V)
            P = 0.95 * P
            print(l)
        
        #Refining our guess of Psat by doing a bisection between Pmin and Pmax 
        while abs(l-1) > tol:
            P = (Pmin + Pmax) * 0.5
            fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
            if l > 1:
                Pmax = P
            else:
                Pmin = P        
        #reporting the values of lsat, Psat and Vsat 
        print("liquid fraction is: {:.4f} and the bubble point pressure = {:.4f}".format(l, P, V))
        Volume = np.divide(Volume, V)
        print(Volume)
        print(Pressure)
        
        #Plotting the Pressure vs Volume curve
        plt.plot(Volume, Pressure, 'r-', label="P(V) with tuning")
    plt.legend()
    plt.show()


# In[55]:


CME_PR_comparison_BinaryCoeffs(False)


# In[56]:


def CME_PR_comparison_Tunning_Molecular_Weight(ChangeWeight):
    #If ChangeWeight is set equal to True a plot will be drawn for the case with an tunned plus fraction
    #(increasing the molecular weight of the plus fraction). The Zi, Tc, Pc and omegas are computed using 
    #the same plus fraction and expansion but with different Mw for the plus fraction (Which results in
    #different coefficient A, B, C and D and therefore for different critical parameters).
    # The inputs of the problem:
    T = 106 + 273.15 # in Kelvin
    Pmin = 1e6      #starting from Pmin
    Pmax = 2e7
    P = Pmax
    R = 8.314 # gas constant

    Names = {'CO2' 'N2' 'C1' 'C2' 'C3' 'i-C4' 'n-C4' 'i-C5' 'n-C5' 'C6' 'C7' 'C8' 'C9' 'C10' 'PS1' 'PS2' 'PS3' 'PS4' 'PS5'}
    
    zi = np.array([0.0044, 0.0017, 0.3463, 0.0263, 0.0335, 0.092, 0.0175, 0.0089, 0.0101, 0.0152, 0.05,                  0.0602, 0.0399, 0.0355, 0.1153, 0.0764, 0.0633, 0.0533, 0.0330]) # overall composition
    Tc = np.array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4, 469.6, 507.4, 548,                  575, 603, 626, 633.1803, 675.9365, 721.3435, 785.0532, 923.8101]) # in Kelvin
    Pc = np.array([72.9, 33.6, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3, 29.3, 30.7, 28.4, 26,                   23.9, 21.6722, 19.0339, 16.9562, 14.9613, 12.6979])*101325 # in Pa
    omega  = np.array([0.228, 0.04, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227, 0.251, 0.296, 0.28,                  0.312, 0.348, 0.385, 0.6254, 0.7964, 0.9805, 1.2222, 1.4000]) # accentric factors
    Nc = zi.size # number of components
    Ki = Ki_guess(Pc, Tc, Pmax, T, omega, Nc)  # inital (given) K-values
    
    EOS = 'PRF' # Equation of State we are interested to use, binary coeffs are set equal to zero
    tol  = 1e-5
    conv = 0
    
    l = 0.5
    Volume = []
    Pressure = []
    #finding Pmax that we are going to use for bisection
    while(P > Pmin):
        fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
        if l > 1:
            Pmax = P
        Pressure.append(P)    
        Volume.append(V)
        P = 0.95 * P
        print(l)
     
    while abs(l-1) > tol:
        P = (Pmin + Pmax) * 0.5
        fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
        if l > 1:
            Pmax = P
        else:
            Pmin = P

    print("liquid fraction is: {:.4f} and the bubble point pressure = {:.4f}".format(l, P))
    Volume = np.divide(Volume, V)
    print(Volume)
    print(Pressure)
    
    #Plotting the Pressure vs Volume curve
    plt.plot(Volume, Pressure, 'b-', label="P(V) without tunning")
    plt.xlim(0.9,1.4)
    plt.ylim(0.6e7, 2e7)
    plt.xlabel("Relative Volume")
    plt.ylabel("Pressure (Pa)")
    plt.title("Pressure against volume")
    plt.legend()

    if ChangeWeight:
        EOS = 'PRF' # Equation of State we are interested to use, binary coeffs are set equal to zero
        
        tol  = 1e-5
        #new input parameters correspoding to the new molecular weight
        zi = np.array([0.0044, 0.0017, 0.3463, 0.0263, 0.0335, 0.092, 0.0175, 0.0089, 0.0101, 0.0152, 0.05,                      0.0602, 0.0399, 0.0355, 0.0802, 0.0613, 0.0600, 0.0650, 0.0749]) # overall composition
        Tc = np.array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4, 469.6, 507.4, 548,                      575, 603, 626, 631.7034, 671.9903, 715.4580, 777.8797, 965.5452]) # in Kelvin
        Pc = np.array([72.9, 33.6, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3, 29.3, 30.7, 28.4, 26,                       23.9, 21.6590, 19.0496, 16.9786, 14.9686, 12.3169])*101325 # in Pa
        omega  = np.array([0.228, 0.04, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227, 0.251, 0.296, 0.28,                      0.312, 0.348, 0.385, 0.6272, 0.7978, 0.9829, 1.2289, 1.2296]) # accentric factors
        P = 2e7
        Pmin = 1e6 
        l = 0.5
        Volume = []
        Pressure = []
        #finding Pmax that we are going to use for bisection
        while(P > Pmin):
            fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
            if l > 1:
                Pmax = P
            Pressure.append(P)    
            Volume.append(V)
            P = 0.95 * P
            print(l)
            
        while abs(l-1) > tol:
            P = (Pmin + Pmax) * 0.5
            fug_v, fug_l, l, V = flash(EOS, 0.5, Nc, zi, Tc, Pc, P, T, omega)
            if l > 1:
                Pmax = P
            else:
                Pmin = P        
            
        print("liquid fraction is: {:.4f} and the bubble point pressure = {:.4f}".format(l, P, V))
        Volume = np.divide(Volume, V)
        print(Volume)
        print(Pressure)
        
        #Plotting the Pressure vs Volume curve
        plt.plot(Volume, Pressure, 'r-', label="P(V) with tuning")
    plt.legend()
    plt.show()


# In[57]:


CME_PR_comparison_Tunning_Molecular_Weight(True)


# In[58]:


CME_PR_comparison_Tunning_Molecular_Weight(False)


# In[ ]:




