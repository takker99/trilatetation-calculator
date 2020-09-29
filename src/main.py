import numpy as np
import math


def calccos(oa: float, ob: float, ab: float):
    return (oa ** 2 + ob ** 2 - ab ** 2) / (2 * oa * ob)


def makeC(l: list):
    # l[0] = oa
    # l[1] = ob
    # l[2] = oc
    # l[3] = ab
    # l[4] = bc
    # l[5] = ca
    sin0 = math.sqrt(1-calccos(l[2], l[0], l[5])**2) # COA
    sin1 = math.sqrt(1-calccos(l[0], l[1], l[3])**2) # AOB
    sin2 = math.sqrt(1-calccos(l[1], l[2], l[4])**2) # BOC
    c = [
        -(-l[2]**2+l[5]**2+l[0]**2)/(2*l[2]*l[0]**2*sin0) +
        (l[0]**2+l[3]**2-l[1]**2)/(2*l[0]**2*l[1]*sin1),
        (-l[0]**2+l[3]**2+l[1]**2)/(2*l[0]*l[1]**2*sin1) +
        (l[1]**2+l[4]**2-l[2]**2)/(2*l[1]**2*l[2]*sin2),
        (-l[1]**2+l[4]**2+l[2]**2)/(2*l[1]*l[2]**2*sin2) -
        (l[2]**2+l[5]**2-l[0]**2)/(2*l[2]**2*l[0]*sin0),
        -l[3]/(l[0]*l[1]*sin1),
        -l[4]/(l[1]*l[2]*sin2),
        - l[5] / (l[2] * l[0] * sin0)
    ]
    print(f'c =\n{c}')
    return np.matrix(c).T


def makeOmega(l: list):
    theta0 = math.acos(calccos(l[2], l[0], l[5]))
    theta1 = math.acos(calccos(l[0], l[1], l[3]))
    theta2 = math.acos(calccos(l[1], l[2], l[4]))
    return theta0 - theta1 - theta2


# 繰り返す函数
def calcResidual(lengths: np.matrix):
    c = makeC(*lengths.T.tolist())
    omega = makeOmega(*lengths.T.tolist())
    return -(omega/sum([i**2 for i in c.T.tolist()[0]]))*c

# 収束判定


def checkConvergence(residuals: np.matrix):
    value = max([max([math.fabs(residual) for residual in row])
                 for row in residuals])
    print(f'value = {value}')
    return value < 10.0**-6 # μmまで合えば良し


# main函数
if __name__ == '__main__':
    # 初期値
    lengths = np.matrix([
        [91.882],
        [97.682],
        [63.419],
        [75.814],
        [90.791],
        [129.469],
    ], dtype=float)

    i = 0
    print(f'initial lengths:\n{lengths}')
    residual = calcResidual(lengths)
    # 残差を保持するlist
    residuals = residual.T.tolist()
    lengths += residual
    i += 1
    print(f'step = {i}, lengths =\n{lengths}')
    print(f'residual =\n{residual}')

    # 計算処理
    while not checkConvergence(residuals):
        residual = calcResidual(lengths)
        residuals.append(*residual.T.tolist())
        lengths += residual
        i += 1
        print(f'step = {i}, lengths =\n{lengths}')
        print(f'residual =\n{residual}')
        if len(residuals) > 10:
            residuals.pop(0)

    print(
        f'Finish calculating!\n\tMPV: {lengths}\n\tresidual: {residuals[len(residuals)-1]}')
