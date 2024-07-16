import numpy as np

def SIR(gamma,beta,S0,I0,tf,Num,r_ruido):
    N = S0 + I0
    t = np.linspace(0,tf,Num)
    dt = t[1] - t[0]
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0

    for i in range(len(t)-1):
        dS = -beta*I[i]*S[i]/N
        dI = beta*I[i]*S[i]/N - gamma*I[i]
        dR = gamma*I[i]

        S[i+1] = S[i] + dS*dt
        I[i+1] = I[i] + dI*dt
        R[i+1] = R[i] + dR*dt

    #t = t + np.random.uniform(-1, 1, size=len(t))
    S = S + N*np.random.normal(0, r_ruido, size=len(t))
    I = I + N*np.random.normal(0, r_ruido, size=len(t))
    R = R + N*np.random.normal(0, r_ruido, size=len(t))

    #S = np.array([S[i]*(np.random.uniform(-r_ruido,r_ruido)+1) for i in range(len(S))])
    #I = np.array([I[i]*(np.random.uniform(-r_ruido,r_ruido)+1) for i in range(len(I))])
    #R = np.array([R[i]*(np.random.uniform(-r_ruido,r_ruido)+1) for i in range(len(R))])

    #S = np.array([S[i]*(np.random.normal(0,r_ruido)+1) for i in range(len(S))])
    #I = np.array([I[i]*(np.random.normal(0,r_ruido)+1) for i in range(len(I))])
    #R = np.array([R[i]*(np.random.normal(0,r_ruido)+1) for i in range(len(R))])

    return list(np.abs(S)),list(np.abs(I)),list(np.abs(R))  #,list(t) 

def SIR_Estocastico(gamma,beta,S0,I0,tf,Num):
        N = S0 + I0
        t = np.linspace(0,tf,Num)
        dt = t[1] - t[0]
        
        N = S0 + I0

        S = np.zeros(Num)
        I = np.zeros(Num)
        R = np.zeros(Num)

        S[0] = S0
        I[0] = I0


        for i in range(len(t)-1):
            S_I = np.random.binomial(S[i], beta*I[i]/N * dt)
            I_R = np.random.binomial(I[i], gamma*dt)
            
            S[i+1] = S[i] - S_I
            I[i+1] = I[i] + S_I - I_R
            R[i+1] = R[i] + I_R

        return list(np.abs(S)),list(np.abs(I)),list(np.abs(R))  

def SIR_simulator(n=100):
    Num = 100
    
    S_total = np.zeros([n,Num])
    I_total = np.zeros([n,Num])
    R_total = np.zeros([n,Num])
    gamma_total = np.zeros(n)
    Beta_total = np.zeros(n)

    for i in range(n):
        gamma = np.abs(np.random.normal(0.05,0.05))
        Beta = np.abs(np.random.normal(0.75,1))
        Personas = np.abs(np.random.normal(2500,1000))
        #r_ruido = np.abs(np.random.normal(0,0.1))
        r_ruido = np.abs(np.random.uniform(0.02,0.03))
        S,I,R = SIR(gamma,Beta,Personas,1,100,Num,r_ruido)
        #S,I,R = SIR_Estocastico(gamma,Beta,Personas,1,100,Num)

        S_total[i,:] = S
        I_total[i,:] = I
        R_total[i,:] = R
        gamma_total[i] = gamma
        Beta_total[i] = Beta
        

    return S_total,I_total,R_total,gamma_total,Beta_total


S_total,I_total,R_total,gamma_total,Beta_total = SIR_simulator(1000000)

threshold = 1e+4  # Adjust according to your specific case

large_S_indices = np.abs(S_total) > threshold
large_I_indices = np.abs(I_total) > threshold
large_R_indices = np.abs(R_total) > threshold

problematic_indices = np.any(large_S_indices, axis=1) | np.any(large_I_indices, axis=1) | np.any(large_R_indices, axis=1)

S_total = S_total[~problematic_indices]
I_total = I_total[~problematic_indices]
R_total = R_total[~problematic_indices]
Gamma_filtered = [gamma_total[i] for i in range(len(gamma_total)) if not problematic_indices[i]]
Beta_filtered = [Beta_total[i] for i in range(len(Beta_total)) if not problematic_indices[i]]

np.savez("DataSetSIR7.npz", S_=S_total, I_=I_total, R_=R_total, gamma_=Gamma_filtered, Beta_=Beta_filtered)