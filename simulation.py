import numpy as np
import time
from decimal import Decimal


def initialization(N, n):
    x_k = np.linspace(0, 1, N+1)
    
    Psi_real = np.zeros(N+1)
    Psi_imag = np.zeros(N+1)
    
    for i,x in enumerate(x_k):
        Psi_real[i] = np.sqrt(2) * np.sin(n * np.pi * x)
    
    return Psi_real, Psi_imag


def hamiltonian(tab, N, K, Omega, tau):
    tmp = np.zeros(N+1)
    
    for i in range(1,N):
        tmp[i] = -(1/2)*(tab[i+1] + tab[i-1] - 2*tab[i])/np.power(1/N,2) + K*( (i/N) - (1/2) )*tab[i]*np.sin(Omega * tau)
    
    return tmp


def sim(Psi_real, Psi_imag, N_sim, S_out, S_dat, dtau, N, K, Omega):
    tau = 0
    
    with open('sim.out', 'w+') as f1, open('sim.dat', 'w+') as f2:
        f1.write('t' + '\t' + 'N_out' + '\t' + 'x_out' + '\t' + 'epsilon_out' + '\n')
    
        for i in range(N_sim):
            Psi_real = Psi_real + hamiltonian(Psi_imag, N, K, Omega, tau)*(dtau/2)
            tau = tau + (dtau/2)

            Psi_imag = Psi_imag - hamiltonian(Psi_real, N, K, Omega, tau)*dtau
            tau = tau + (dtau/2)
            
            tmp = hamiltonian(Psi_imag, N, K, Omega, tau)
            Psi_real = Psi_real + tmp*(dtau/2)
         
            if i%S_out == 0:    
                N_out = np.sum(Psi_imag**2 + Psi_real**2)/N
                x_out = np.sum( np.linspace(0, 1, N+1)*(Psi_imag**2 + Psi_real**2) )/N
                epsilon_out = ( np.dot(Psi_imag, tmp) + np.dot(Psi_real, hamiltonian(Psi_real, N, K, Omega, tau)) )/N
            
                f1.write("{:.5f}".format(tau) + '\t')
                #f1.write("{:.6E}".format(Decimal(N_out)) + '\t')
                #f1.write("{:.6E}".format(Decimal(x_out)) + '\t')
                #f1.write("{:.6E}".format(Decimal(epsilon_out)) + '\t')
                f1.write("{}".format(N_out) + '\t')
                f1.write("{}".format(x_out) + '\t')
                f1.write("{}".format(epsilon_out) + '\t')
                f1.write('\n')
                
            if i%S_dat == 0:    
                rho = Psi_real[0:N+1:2]**2 + Psi_imag[0:N+1:2]**2

                f2.write('\t'.join(map(str, rho)))
                f2.write('\n')


if __name__ == "__main__":
    start_time = time.time()
    
    file = open('parameters.input')
    params = {}
    for line in file:
        line = line.strip()
        key_value = line.split('#')
        if len(key_value) == 2:
            params[key_value[1].strip()] = np.float(key_value[0].strip())
    

    Psi_real, Psi_imag = initialization(int(params['N']), int(params['n']))
    sim(Psi_real, Psi_imag, int(params['N_sim']), int(params['S_out']), int(params['S_dat']),
        float(params['dtau']), int(params['N']), float(params['K']), float(params['Omega']))

    print("Simulation took", round(time.time() - start_time,2) , "s to run")