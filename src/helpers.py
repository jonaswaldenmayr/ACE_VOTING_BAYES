


import math


def E_SCC_reduction_function(x: float, E_SCC_lowerbound: float = 0.0, E_SCC_Init: float = 1.0,
               rate: float = 1.0, midpoint: float=0) -> float:
    return E_SCC_lowerbound + (E_SCC_Init - E_SCC_lowerbound) / (1 + math.exp(rate * (x - ((midpoint)))))

def pol_system_cal(t:int) -> float:
        #return t*(1/2.5)
        return 1


