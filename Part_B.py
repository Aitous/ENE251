#!/usr/bin/env python
# coding: utf-8

# In[74]:


import time
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression


# In[ ]:





# In[75]:


# Solving RachfordRice

def SolveRachfordRice(l, Nc, z, K):

    F  = lambda l: sum([(1 - K[i]) * z[i]/(K[i] + (1 - K[i]) * l) for i in range(Nc)])
    dF = lambda l: sum([-z[i] * (1 - K[i])**2/((K[i] + (1 - K[i]) * l)**2) for i in range(Nc)])
    
    #F0 = F(0)
    #F1 = F(1)
    #
    #if(F0 > 0 and F1 < 0):
    #    lmin = 0
    #    lmax = 1
    #elif(F0 > 0 and F1 > 0):
    #    lmin = 1
    #    lmax = np.max([(K[i]*z[i] - K[i])/(1 - K[i]) for i in range(Nc)])
    #    
    #else:
    #    lmax = 0
    #    lmin = np.min([(z[i] - K[i])/(1 - K[i]) for i in range(Nc)])
   #
    useNewton = True                                #Change to false for bisection only
    #error = []                                      #error array
    #i = 0
    ##l = (lmin + lmax)*0.5
    #while abs(F(l)) > 1.e-5:
    #    if(F(l) > 0):
    #        lmin = l
    #    else: 
    #        lmax = l
    #    delta_l = - F(l) / dF(l)
    #    if(l + delta_l > lmin and l + delta_l < lmax and useNewton):
    #        l = l + delta_l
    #    else:
    #        l = 0.5 * (lmin + lmax)
    #    error.append(F(l))
    #    #print('error = ', error[i])                 #reporting error for each step
    #    i += 1
    #    
    #return l

    lmin = 0.0
    lmax = 1.0
    # Corner case
    if F(1.0) > 0:
        lmin = 1.0
        kmax = max(K)
        lmax = -kmax / (1.0-kmax)
    elif F(0.0) < 0:
        kmin = min(K)
        lmin = - kmin / (1.0-kmin)
        lmax = 0.0

    curr_F = F(l)

    while abs(curr_F) > 1.0e-8:
        if curr_F > 0:
            lmin = l
        else:
            lmax = l
        newton_update = - curr_F / dF(l)
        l_newton = l + newton_update
        if lmin < l_newton < lmax and useNewton: # Newton is valid
            l = l_newton
        else: # use Bisection
            l = 0.5*(lmin + lmax)

        curr_F = F(l)
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
    
    return (fug_v, fug_l, l, xi, yi)

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
    return V


# In[76]:


def MMP_determination(Zoil, Zinj, makePlot = True):
    #This is the function determining the MMP. It implements the mixing cell method.
    #The incrementation of the pressure is dynamically handled by first adding a 200 psia to the initial pressure
    #to determine the second pressure and for the third one we do an linear extrapolation to find the MMP estimate 
    #and we update the pressure by adding the third (or any choosing fraciton) of the pressure difference between 
    #current pressure and MMP. For subsquent pressures I tried using regression like in the article. And I update
    #the pressure based on the MMP estimate from the regression.
    P = 500 * 6894.76
    T = 106 + 273.15
    
    R = 8.314 # gas constant
    
    Names = {'CO2' 'N2' 'C1' 'C2' 'C3' 'iC4' 'n-C4' 'i-C5' 'n-C5' 'C6' 'C7' 'C8' 'C9' 'C10' 'PS1' 'PS2' 'PS3' 'PS4' 'PS5'}

    Tc = np.array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4, 469.6, 507.4, 548,                  575, 603, 626, 633.1803, 675.9365, 721.3435, 785.0532, 923.8101]) # in Kelvin
    Pc = np.array([72.9, 33.6, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3, 29.3, 30.7, 28.4, 26,                   23.9, 21.6722, 19.0339, 16.9562, 14.9613, 12.6979])*101325 # in Pa
    omega  = np.array([0.228, 0.04, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227, 0.251, 0.296, 0.28,                  0.312, 0.348, 0.385, 0.6254, 0.7964, 0.9805, 1.2222, 1.4000]) # accentric factors

    EOS = 'PR' # Equation of State we are interested to use
    Nc = len(Zinj)
    
    TLmin = []    #array containing the values of the min Tie-line length for each pressure step.
    TLoil = []    #array containing the values of the oil Tie-line length for each pressure step.
    TLgas = []    #array containing the values of the gas Tie-line length for each pressure step.
    Pressure = []
    K = []
    alpha = 0.5
    
    tol = 1e-2
    tolMMP = 50 * 6894.76 #pressure tolerance
    deltaMMP = P
    MMP = P
    step = 1      #Pressure step.
    maxContact = 30
    
    while deltaMMP >= tolMMP:
        Ncont = 1
        comp = [[Zinj, Zoil]]
        tieLength = []
        while Ncont < maxContact:
            cells = comp[- 1]
            tempComp = np.zeros((2*Ncont + 2, Nc))
            tempTie  = []
            K_value = []
            tempComp[0] = Zinj
            #Start mixing cells for this contact number.
            for i in range(0, len(cells), 2):
                Zmix = cells[i+1] + alpha * (cells[i] - cells[i+1])
                fug_v, fug_l, l, xi, yi = flash(EOS, 0.5, Nc, Zmix, Tc, Pc, P, T, omega)
                tempTie.append(np.sqrt(np.sum(np.square(yi - xi))))
                K_value.append(yi/xi)
                tempComp[i+1] = xi
                tempComp[i+2] = yi

            
            tempComp[-1] = Zoil
            comp.append(tempComp)
            tieLength.append(tempTie)
            K.append(K_value)
            #checking for the tie length convergence: if three consecutive cells between two successive contacts
            #have the same Tie-line Length to a certain tolerance.
            if Ncont >= 3 and len(tieLength[Ncont-2])>= 3:
                count = 0
                if abs(tieLength[Ncont-1][0] - tieLength[Ncont-2][0]) <= tol                          and abs(tieLength[Ncont-1][1] - tieLength[Ncont-2][1]) <= tol                                 and abs(tieLength[Ncont-1][2] - tieLength[Ncont-2][2])<= tol:
                    count += 1
                n = len(tieLength[Ncont-2])//2
                if abs(tieLength[Ncont-1][n-1] - tieLength[Ncont-2][n-1]) <= tol                         and abs(tieLength[Ncont-1][n] - tieLength[Ncont-2][n]) <= tol                             and abs(tieLength[Ncont-1][n+1] - tieLength[Ncont-2][n+1])<= tol:
                    count += 1
                if abs(tieLength[Ncont-1][-1] - tieLength[Ncont-2][-1]) <= tol                         and abs(tieLength[Ncont-1][-2] - tieLength[Ncont-2][-2]) <= tol                             and abs(tieLength[Ncont-1][-3] - tieLength[Ncont-2][-3])<= tol:
                    count += 1
                if count == 3: 
                    break
            Ncont += 1
        if step >= 2 and np.min(tempTie) >= TLmin[-1]:
            break
        Pressure.append(P)
        TLmin.append(np.min(tempTie))
        TLgas.append(tempTie[0])     
        TLoil.append(np.max(tempTie))
        step += 1
        #updating the pressure.
        if step == 2:
            P = P + 200 * 6894.76
        if step >= 3:
            slope = np.divide(TLmin[-2] - TLmin[-1], Pressure[-2] - Pressure[-1])
            b = TLmin[-1]- slope * Pressure[-1]
            deltaMMP = -np.divide(b,slope) - P
            MMP = -np.divide(b,slope)
            if MMP < 0:
                break;
            P = P + (MMP - P)/3
        if step >= 4:              #Extrapolating the next MMP estimation using a regression
            fitting = False
            exponent = 1.5
            while exponent <= 10:  #I didn't find any function doing the the regression for the exponent, so
                                   #So, I am doing a while loop on the exponent and regressing on the a and b.
                y = np.array(Pressure[-3:])
                x = np.array(np.power(TLmin[-3:], exponent)).reshape(-1,1)
                model = LinearRegression().fit(x,y)
                if model.score(x, y) > 0.999:
                    deltaMMP = model.intercept_ - P
                    MMP = model.intercept_
                    fitting = True
                    break
                exponent += 0.1
            if fitting == False:
                slope = np.divide(TLmin[-2] - TLmin[-1], Pressure[-2] - Pressure[-1])
                b = TLmin[-1] - slope * Pressure[-1]
                deltaMMP = -np.divide(b,slope) - P
                MMP = -np.divide(b,slope)
            P = P + (MMP - P)/3
        if abs(TLmin[-1] <= 1e-5):
            break
        print("Number of contact: ", Ncont)
        print("Converged pressure at this step:", P)
        print("MMP pressure estimate:", MMP)
        print("---")
    
    # plotting tie-line Length vs Cell number
    if makePlot:
        plt.plot(Pressure, TLmin, 'rs', label='Crossover')
        plt.plot(Pressure, TLoil, 'b^', label='Oil')
        plt.plot(Pressure, TLgas, 'g+', label='Gas')
        plt.xlabel("Pressure (Pa)")
        plt.ylabel("Tie-line Length")
        plt.title("Development of key Tie-Lines")
        plt.legend(loc='best')
        plt.show()
    
    return P


# In[77]:


def MMP_vs_Composition(LPG_CO2_comb, makePlot = True):
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
    MMPressures = []
    composition = []
    
    Oilcomp = np.array([0.0044, 0.0017, 0.3463, 0.0263, 0.0335, 0.092, 0.0175, 0.0089, 0.0101, 0.0152, 0.05,                  0.0602, 0.0399, 0.0355, 0.1153, 0.0764, 0.0633, 0.0533, 0.0330])#oil composition
    
    for i in range(numMixtures):
        Injcomp = np.array(LPG_CO2_comb[i] * LPG + (1 - LPG_CO2_comb[i]) * CO2)
        MMPressures.append(MMP_determination(Oilcomp, Injcomp, True))
        composition.append(LPG_CO2_comb[i])

    plt.plot(composition, MMPressures)
    plt.xlabel('Composition')        
    plt.ylabel('MMPs (Pa)')
    plt.title('MMPs vs compositions')
    plt.show()
    

            


# In[79]:


MMP_vs_Composition(np.array([0.7, 0.55, 0.4, 0.2, 0.1, 0.]), True)


# In[ ]:




