import numpy as np
import pandas as pd
import math
from typing import Sequence
import argparse


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
    if len(residuals) == 0: return False
    value = max([max([math.fabs(residual) for residual in row])
                 for row in residuals])
    return value < 10.0**-6 # μmまで合えば良し

def printLengths(lengths: Sequence[float], indent: int = 1, symbol: str = 'L'):
    space = ''.join(['\t']*indent)
    for i in range(len(lengths)):
        print(f'{space}{symbol}_{i} = {lengths[i]:.6g}m')


def printStepLengths(lengths: Sequence[float], step: int, indent: int = 1, symbol: str = 'l'):
    space = ''.join(['\t']*indent)
    for i in range(len(lengths)):
        print(f'{space}{symbol}_{i}({step}) = {lengths[i]:.6g}m')


# main函数
if __name__ == '__main__':
    # commald-line argumentsの設定
    parser = argparse.ArgumentParser(
        description='三辺測量によって得た四辺形の6個の辺長の最確値と残差を求めるscript')
    parser.add_argument('input_file', help='辺長の値が書き込まれたcsvファイル')
    parser.add_argument('--ignore', '-I', choices=[
                        'header', 'index', 'both'], default='none', help='csv fileのheader及びindexを無視するかどうか')
    args = parser.parse_args()

    df=None

    # fileを読み込む
    if args.ignore == 'header':
        df = pd.read_csv(args.input_file, usecols=[0])
    elif args.ignore == 'index':
        df = pd.read_csv(args.input_file, index_col=0, header=None, usecols=[0, 1])
    elif args.ignore == 'both':
        df = pd.read_csv(args.input_file, index_col=0, usecols=[0, 1])
    else:
        df = pd.read_csv(args.input_file, header=None, usecols=[0])

    # 初期値
    measured_lengths = np.matrix(df, dtype=float)
    lengths = np.matrix(measured_lengths);

    i = 0
    print('This program calculates lengths in metre.')
    print('Measured lengths:')
    printStepLengths(*lengths.T.tolist(), i, 1)
    print('Start calculating...')
    # 残差を保持するlist
    residuals = []

    # 計算処理
    while not checkConvergence(residuals):
        residual = calcResidual(lengths)
        residuals.append(*residual.T.tolist())
        lengths += residual
        i += 1
        print(f'### result of step {i} ###')
        printStepLengths(*lengths.T.tolist(), i)
        printStepLengths(*residual.T.tolist(), i-1, symbol='⊿')
        if len(residuals) > 10:
            residuals.pop(0)
    print(f'##########################')
    print('Finish calculating!')
    print('Most probable lengths:')
    printLengths(*measured_lengths.T.tolist())
    print('Residuals:')
    printLengths(*(lengths - measured_lengths).T.tolist(), symbol='ν')
