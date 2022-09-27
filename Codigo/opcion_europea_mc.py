import numpy as np
import math

"""
opcion_europea_mc
Def
    Calculador del precio de una opcion Europea con el modelo de MonteCarlo
Inputs
    - tipo : string - Tipo de contrato entre ["CALL","PUT"]
    - S : float - Spot price del activo
    - K : float - Strike price del contrato
    - T : float - Tiempo hasta la expiracion (en años)
    - r : float - Tasa 'libre de riesgo' (anualizada)
    - sigma : float - Volatilidad implicita (anualizada)
    - div : float - Tasa de dividendos continuos (anualizada)
    - pasos : int - Cantidad de caminos de montecarlo
Outputs
    - precio_MC: float - Precio del contrato
"""

def opcion_europea_mc(tipo, S, K, T, r, sigma, div, pasos):

    z = np.random.normal(0,1,pasos)
    opcion = np.zeros(pasos)
    for i in range(0,pasos):
        if tipo == "C":
            payoff = max( 0 , S * math.exp((r-div - 0.5 * math.pow(sigma,2)) * T + sigma * math.sqrt(T)  * z[i]) - K)
        elif tipo == "P":
            payoff = max(0, K - S * math.exp((r-div - 0.5 * math.pow(sigma, 2)) * T + sigma * math.sqrt(T) * z[i]) )

        opcion[i] = math.exp(-r * T) * payoff

    precio_MC = np.mean(opcion)

    #var= np.var(opcion)

    return precio_MC


def correlated_sampler(N, M, rho):
    mu = np.array([0,0])
    cov = np.array([[1,rho],
                    [rho,1]])
    Z = np.random.multivariate_normal(mu, cov, (N,M))

    return Z


def opcion_europea_mc_heston(tipo='C', S=400.38, K=400.38, T=1, r=0.0329, v0=0.043081, div=0.01, pasos=252):
    """
        v0 :    float - volatilidad inicial
        pasos : int -   cantidad de pasos de tiempo
        M :     int -   cantidad de simulaciones
        rho :   float - correlacion de los procesos de wiener
        theta : float - valor medio de la volatilidad
        kappa : float - velocidad de retorno a la media
        sigma : float - desvio estandard de la volatilidad
    """

    theta = 0.110948 
    kappa = 1.658242 
    sigma = 1.000000 
    rho = -0.520333 

    if pasos > 500:
        pasos = 500

    M = pasos * 80           # cantidad de escenarios/simulaciones

    dt = T/pasos

    __S = np.full(shape=(pasos+1,M), fill_value=S)
    v = np.full(shape=(pasos+1,M), fill_value=v0)

    Z = correlated_sampler(pasos, M, rho)

    for i in range(1,pasos+1):
        __S[i] = __S[i-1] * np.exp((r - div - 0.5*v[i-1])*dt + np.sqrt(v[i-1] * dt) * Z[i-1,:,0])
        v[i] = np.maximum(v[i-1] + kappa*(theta-v[i-1])*dt + sigma*np.sqrt(v[i-1]*dt)*Z[i-1,:,1],0)

    c = np.exp(-r*T)*np.mean(np.maximum(__S-K,0))
    p = np.exp(-r*T)*np.mean(np.maximum(K-__S,0))
    if tipo == 'C':
        return c
    else:
        return p


c = opcion_europea_mc_heston(tipo='C', pasos=1000)
print(c)
c = opcion_europea_mc(tipo='C', S=400.38, K=400.38, T=1, r=0.0329, sigma=0.14, div=0.01, pasos=10000)
print(c)
