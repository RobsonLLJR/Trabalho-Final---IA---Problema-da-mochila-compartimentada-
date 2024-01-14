from random import Random
from time import time
from inspyred import ec
from inspyred.ec import terminators
import numpy as np

def gera_populacao(random, args):
    size = args.get('num_inputs', 48)
    return  [random.randint(0, 456) for i in range(size)]

def avaliacao(candidates, args):
    fitness = []
    for cs in candidates:
        fit = calcula_fitness(cs[0], cs[1], cs[2], cs[3], cs[4], cs[5], cs[6], cs[7], cs[8], cs[9], cs[10], cs[11], cs[12], cs[13], cs[14], cs[15],
                              cs[16], cs[17], cs[18], cs[19], cs[20], cs[21], cs[22], cs[23], cs[24], cs[25], cs[26], cs[27], cs[28], cs[29], cs[30],
                              cs[31], cs[32], cs[33], cs[34], cs[35], cs[36], cs[37], cs[38], cs[39], cs[40], cs[41], cs[42], cs[43], cs[44], cs[45],
                              cs[46], cs[47])
        fitness.append(fit)
    return fitness

def calcula_fitness(c1_T1, c1_F1, c1_L1, c1_T2, c1_F2, c1_L2, c2_T1, c2_F1, c2_L1, c2_T2, c2_F2, c2_L2, c2_T3, c2_F3, c2_L3, c2_T4, c2_F4, c2_L4,
                    c3_T1, c3_F1, c3_L1, c3_T2, c3_F2, c3_L2, c3_T3, c3_F3, c3_L3, c3_T4, c3_F4, c3_L4, c4_T1, c4_F1, c4_L1, c4_T2, c4_F2, c4_L2,
                    c4_T3, c4_F3, c4_L3, c4_T4, c4_F4, c4_L4, c4_T5, c4_F5, c4_L5, c4_T6, c4_F6, c4_L6):

    c1_T1 = np.round(c1_T1)
    c1_F1 = np.round(c1_F1)
    c1_L1 = np.round(c1_L1)
    c1_T2 = np.round(c1_T2)
    c1_F2 = np.round(c1_F2)
    c1_L2 = np.round(c1_L2)
    c2_T1 = np.round(c2_T1)
    c2_F1 = np.round(c2_F1)
    c2_L1 = np.round(c2_L1)
    c2_T2 = np.round(c2_T2)
    c2_F2 = np.round(c2_F2)
    c2_L2 = np.round(c2_L2)
    c2_T3 = np.round(c2_T3)
    c2_F3 = np.round(c2_F3)
    c2_L3 = np.round(c2_L3)
    c2_T4 = np.round(c2_T4)
    c2_F4 = np.round(c2_F4)
    c2_L4 = np.round(c2_L4)
    c3_T1 = np.round(c3_T1)
    c3_F1 = np.round(c3_F1)
    c3_L1 = np.round(c3_L1)
    c3_T2 = np.round(c3_T2)
    c3_F2 = np.round(c3_F2)
    c3_L2 = np.round(c3_L2)
    c3_T3 = np.round(c3_T3)
    c3_F3 = np.round(c3_F3)
    c3_L3 = np.round(c3_L3)
    c3_T4 = np.round(c3_T4)
    c3_F4 = np.round(c3_F4)
    c3_L4 = np.round(c3_L4)
    c4_T1 = np.round(c4_T1)
    c4_F1 = np.round(c4_F1)
    c4_L1 = np.round(c4_L1)
    c4_T2 = np.round(c4_T2)
    c4_F2 = np.round(c4_F2)
    c4_L2 = np.round(c4_L2)
    c4_T3 = np.round(c4_T3)
    c4_F3 = np.round(c4_F3)
    c4_L3 = np.round(c4_L3)
    c4_T4 = np.round(c4_T4)
    c4_F4 = np.round(c4_F4)
    c4_L4 = np.round(c4_L4)
    c4_T5 = np.round(c4_T5)
    c4_F5 = np.round(c4_F5)
    c4_L5 = np.round(c4_L5)
    c4_T6 = np.round(c4_T6)
    c4_F6 = np.round(c4_F6)
    c4_L6 = np.round(c4_L6)

    fitness = float((((c1_T1 + c1_F1 + c1_L1) * 28) + ((c1_T2 + c1_F2 + c1_L2) * 30) + ((c2_T1 + c2_F1 + c2_L1) * 7) + ((c2_T2 + c2_F2 + c2_L2) * 26) + ((c2_T3 + c2_F3 + c2_L3) * 30) +
                    ((c2_T4 + c2_F4 + c2_L4) * 27) + ((c3_T1 + c3_F1 + c3_L1) * 8) + ((c3_T2 + c3_F2 + c3_L2) * 29) + ((c3_T3 + c3_F3 + c3_L3) * 10) + ((c3_T4 + c3_F4 + c3_L4) * 35) +
                    ((c4_T1 + c4_F1 + c4_L1) * 23) + ((c4_T2 + c4_F2 + c4_L2) * 12) + ((c4_T3 + c4_F3 + c4_L3) * 21) + ((c4_T4 + c4_F4 + c4_L4) * 9) + ((c4_T5 + c4_F5 + c4_L5) * 26) +
                    ((c4_T6 + c4_F6 + c4_L6) * 33)) / 5637)

    # Restrições peso por compartimento mochila
    h1 = np.maximum(0, float((c1_T1 + c1_T2 + c2_T1 + c2_T2 + c2_T3 + c2_T4 + c3_T1 + c3_T2 + c3_T3 + c3_T4 + c4_T1 + c4_T2 + c4_T3 + c4_T4 + c4_T5 + c4_T6) - 456)) / float(456 / 25)
    h2 = np.maximum(0, float((c1_F1 + c1_F2 + c2_F1 + c2_F2 + c2_F3 + c2_F4 + c3_F1 + c3_F2 + c3_F3 + c3_F4 + c4_F1 + c4_F2 + c4_F3 + c4_F4 + c4_F5 + c4_F6) - 456)) / float(456 / 25)
    h3 = np.maximum(0, float((c1_L1 + c1_L2 + c2_L1 + c2_L2 + c2_L3 + c2_L4 + c3_L1 + c3_L2 + c3_L3 + c3_L4 + c4_L1 + c4_L2 + c4_L3 + c4_L4 + c4_L5 + c4_L6) - 456)) / float(456 / 25)

    # Restrições volume por compartimento mochila

    h4 = np.maximum(0, float((1.5 * (c1_T1 + c1_T2) + (c1_F1 + c1_F2) + (c1_L1 + c1_L2)) - 12))
    h5 = np.maximum(0, float((0.5 * (c2_T1 + c2_T2 + c2_T3 + c2_T4) + (c2_F1 + c2_F2 + c2_F3 + c2_F4) + (c2_L1 + c2_L2 + c2_L3 + c2_L4)) - 12))
    h6 = np.maximum(0, float((2.5 * (c3_T1 + c3_T2 + c3_T3 + c3_T4) + (c3_F1 + c3_F2 + c3_F3 + c3_F4) + (c3_L1 + c3_L2 + c3_L3 + c3_L4)) - 12))


    pesoTotalMochila = 1368

    equilibrio_T = float(456 / pesoTotalMochila)
    equilibrio_F = float(456 / pesoTotalMochila)
    equilibrio_L = float(456 / pesoTotalMochila)

    soma_T = float(c1_T1 + c1_T2 + c2_T1 + c2_T2 + c2_T3 + c2_T4 + c3_T1 + c3_T2 + c3_T3 + c3_T4 + c4_T1 + c4_T2 + c4_T3 + c4_T4 + c4_T5 + c4_T6)
    soma_F = float(c1_F1 + c1_F2 + c2_F1 + c2_F2 + c2_F3 + c2_F4 + c3_F1 + c3_F2 + c3_F3 + c3_F4 + c4_F1 + c4_F2 + c4_F3 + c4_F4 + c4_F5 + c4_F6)
    soma_L = float(c1_L1 + c1_L2 + c2_L1 + c2_L2 + c2_L3 + c2_L4 + c3_L1 + c3_L2 + c3_L3 + c3_L4 + c4_L1 + c4_L2 + c4_L3 + c4_L4 + c4_L5 + c4_L6)
    somaTotal = float(soma_T + soma_F + soma_L)

    #Distribuição de cargas no mochila
    h7 = np.maximum(0, float(((soma_T / somaTotal) - equilibrio_T))) / float(equilibrio_T / 25)
    h8 = np.maximum(0, float(((soma_F / somaTotal) - equilibrio_F))) / float(equilibrio_F / 25)
    h9 = np.maximum(0, float(((soma_L / somaTotal) - equilibrio_L))) / float(equilibrio_L / 25)

    #Restrição de demanda necessária de cada item
    h10 = np.maximum(0, float((c1_T1 + c1_F1 + c1_L1) - 15)) / float(15 / 25)
    h11 = np.maximum(0, float((c1_T2 + c1_F2 + c1_L2) - 16)) / float(16 / 25)
    h12 = np.maximum(0, float((c2_T1 + c2_F1 + c2_L1) - 12)) / float(12 / 25)
    h13 = np.maximum(0, float((c2_T2 + c2_F2 + c2_L2) - 14)) / float(14 / 25)
    h14 = np.maximum(0, float((c2_T3 + c2_F3 + c2_L3) - 20)) / float(20 / 25)
    h15 = np.maximum(0, float((c2_T4 + c2_F4 + c2_L4) - 13)) / float(13 / 25)
    h16 = np.maximum(0, float((c3_T1 + c3_F1 + c3_L1) - 14)) / float(14 / 25)
    h17 = np.maximum(0, float((c3_T2 + c3_F2 + c3_L2) - 10)) / float(10 / 25)
    h18 = np.maximum(0, float((c3_T3 + c3_F3 + c3_L3) - 12)) / float(12 / 25)
    h19 = np.maximum(0, float((c3_T4 + c3_F4 + c3_L4) - 11)) / float(11 / 25)
    h20 = np.maximum(0, float((c4_T1 + c4_F1 + c4_L1) - 21)) / float(21 / 25)
    h21 = np.maximum(0, float((c4_T2 + c4_F2 + c4_L2) - 23)) / float(23 / 25)
    h22 = np.maximum(0, float((c4_T3 + c4_F3 + c4_L3) - 26)) / float(26 / 25)
    h23 = np.maximum(0, float((c4_T4 + c4_F4 + c4_L4) - 21)) / float(21 / 25)
    h24 = np.maximum(0, float((c4_T5 + c4_F5 + c4_L5) - 17)) / float(17 / 25)
    h25 = np.maximum(0, float((c4_T6 + c4_F6 + c4_L6) - 15)) / float(15 / 25)

    fitness = fitness - (h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8 + h9 + h10 + h11 + h12 + h13 + h14 + h15 + h16 + h17 + h18 + h19 + h20 + h21 + h22 + h23 + h24 + h25)

    return fitness

def calculo_solucao(c1_T1, c1_F1, c1_L1, c1_T2, c1_F2, c1_L2, c2_T1, c2_F1, c2_L1, c2_T2, c2_F2, c2_L2, c2_T3, c2_F3, c2_L3, c2_T4, c2_F4, c2_L4,
                    c3_T1, c3_F1, c3_L1, c3_T2, c3_F2, c3_L2, c3_T3, c3_F3, c3_L3, c3_T4, c3_F4, c3_L4, c4_T1, c4_F1, c4_L1, c4_T2, c4_F2, c4_L2,
                    c4_T3, c4_F3, c4_L3, c4_T4, c4_F4, c4_L4, c4_T5, c4_F5, c4_L5, c4_T6, c4_F6, c4_L6):
    c1_T1 = np.round(c1_T1)
    c1_F1 = np.round(c1_F1)
    c1_L1 = np.round(c1_L1)
    c1_T2 = np.round(c1_T2)
    c1_F2 = np.round(c1_F2)
    c1_L2 = np.round(c1_L2)
    c2_T1 = np.round(c2_T1)
    c2_F1 = np.round(c2_F1)
    c2_L1 = np.round(c2_L1)
    c2_T2 = np.round(c2_T2)
    c2_F2 = np.round(c2_F2)
    c2_L2 = np.round(c2_L2)
    c2_T3 = np.round(c2_T3)
    c2_F3 = np.round(c2_F3)
    c2_L3 = np.round(c2_L3)
    c2_T4 = np.round(c2_T4)
    c2_F4 = np.round(c2_F4)
    c2_L4 = np.round(c2_L4)
    c3_T1 = np.round(c3_T1)
    c3_F1 = np.round(c3_F1)
    c3_L1 = np.round(c3_L1)
    c3_T2 = np.round(c3_T2)
    c3_F2 = np.round(c3_F2)
    c3_L2 = np.round(c3_L2)
    c3_T3 = np.round(c3_T3)
    c3_F3 = np.round(c3_F3)
    c3_L3 = np.round(c3_L3)
    c3_T4 = np.round(c3_T4)
    c3_F4 = np.round(c3_F4)
    c3_L4 = np.round(c3_L4)
    c4_T1 = np.round(c4_T1)
    c4_F1 = np.round(c4_F1)
    c4_L1 = np.round(c4_L1)
    c4_T2 = np.round(c4_T2)
    c4_F2 = np.round(c4_F2)
    c4_L2 = np.round(c4_L2)
    c4_T3 = np.round(c4_T3)
    c4_F3 = np.round(c4_F3)
    c4_L3 = np.round(c4_L3)
    c4_T4 = np.round(c4_T4)
    c4_F4 = np.round(c4_F4)
    c4_L4 = np.round(c4_L4)
    c4_T5 = np.round(c4_T5)
    c4_F5 = np.round(c4_F5)
    c4_L5 = np.round(c4_L5)
    c4_T6 = np.round(c4_T6)
    c4_F6 = np.round(c4_F6)
    c4_L6 = np.round(c4_L6)


    print('')
    print('')
    print("PESO IDEAL CLASSE 1")
    print("C1 - Traseiro Item 1:", float(c1_T1))
    print("C1 - Frontal Item 1:", float(c1_F1))
    print("C1 - Lateral Item 1:", float(c1_L1))
    print("C1 - Traseiro Item 2:", float(c1_T2))
    print("C1 - Frontal Item 2:", float(c1_F2))
    print("C1 - Lateral Item 2:", float(c1_L1))
    print("C1 - TOTAL:", float(c1_T1 + c1_T2 + c1_F1 + c1_F2 + c1_L1 + c1_L1 + c1_L2))
    print('')

    print('')
    print('')
    print("PESO IDEAL CLASSE 2")
    print("C2 - Traseiro Item 1:", float(c2_T1))
    print("C2 - Frontal Item 1:", float(c2_F1))
    print("C2 - Lateral Item 1:", float(c2_L1))
    print("C2 - Traseiro Item 2:", float(c1_T2))
    print("C2 - Frontal Item 2:", float(c2_F2))
    print("C2 - Lateral Item 2:", float(c2_L2))
    print("C2 - Traseiro Item 3:", float(c2_T3))
    print("C2 - Frontal Item 3:", float(c2_F3))
    print("C2 - Lateral Item 3:", float(c2_L3))
    print("C2 - Traseiro Item 4:", float(c2_T4))
    print("C2 - Frontal Item 4:", float(c2_F4))
    print("C2 - Lateral Item 4:", float(c2_L4))
    print("C2 - TOTAL Item 1:", float(c2_T1 + c2_F1 + c2_L1))
    print("C2 - TOTAL Item 2:", float(c2_T2 + c2_F2 + c2_L2))
    print("C2 - TOTAL Item 2:", float(c2_T3 + c2_F3 + c2_L3))
    print("C2 - TOTAL Item 2:", float(c2_T4 + c2_F4 + c2_L4))
    print("C2 - TOTAL Classe:", float(c2_T1 + c2_T2 + c2_T3 + c2_T4 + c2_F1 + c2_F2 + c2_F3 + c2_F4 + c2_L1 + c2_L2 + c2_L3 + c2_L4))
    print('')

    print('')
    print('')
    print("PESO IDEAL CLASSE 3")
    print("C3 - Traseiro Item 1:", float(c3_T1))
    print("C3 - Frontal Item 1:", float(c3_F1))
    print("C3 - Lateral Item 1:", float(c3_L1))
    print("C3 - Traseiro Item 2:", float(c3_T2))
    print("C3 - Frontal Item 2:", float(c3_F2))
    print("C3 - Lateral Item 2:", float(c3_L2))
    print("C3 - Traseiro Item 3:", float(c3_T3))
    print("C3 - Frontal Item 3:", float(c3_F3))
    print("C3 - Lateral Item 3:", float(c3_L3))
    print("C3 - Traseiro Item 4:", float(c3_T4))
    print("C3 - Frontal Item 4:", float(c3_F4))
    print("C3 - Lateral Item 4:", float(c3_L4))
    print("C3 - TOTAL Item 1:", float(c3_T1 + c3_F1 + c3_L1))
    print("C3 - TOTAL Item 2:", float(c3_T2 + c3_F2 + c3_L2))
    print("C3 - TOTAL Item 3:", float(c3_T3 + c3_F3 + c3_L3))
    print("C3 - TOTAL Item 4:", float(c3_T4 + c3_F4 + c3_L4))
    print("C3 - TOTAL Classe:", float(c3_T1 + c3_T2 + c3_T3 + c3_T4 + c3_F1 + c3_F2 + c3_F3 + c3_F4 + c3_L1 + c3_L2 + c3_L3 + c3_L4))
    print('')

    print('')
    print('')
    print("PESO IDEAL CLASSE 4")
    print("C4 - Traseiro Item 1:", float(c4_T1))
    print("C4 - Frontal Item 1:", float(c4_F1))
    print("C4 - Lateral Item 1:", float(c4_L1))
    print("C4 - Traseiro Item 2:", float(c4_T2))
    print("C4 - Frontal Item 2:", float(c4_F2))
    print("C4 - Lateral Item 2:", float(c4_L2))
    print("C4 - Traseiro Item 3:", float(c4_T3))
    print("C4 - Frontal Item 3:", float(c4_F3))
    print("C4 - Lateral Item 3:", float(c4_L3))
    print("C4 - Traseiro Item 4:", float(c4_T4))
    print("C4 - Frontal Item 4:", float(c4_F4))
    print("C4 - Lateral Item 4:", float(c4_L4))
    print("C4 - Traseiro Item 5:", float(c4_T5))
    print("C4 - Frontal Item 5:", float(c4_F5))
    print("C4 - Lateral Item 5:", float(c4_L5))
    print("C4 - Traseiro Item 6:", float(c4_T6))
    print("C4 - Frontal Item 6:", float(c4_F6))
    print("C4 - Lateral Item 6:", float(c4_L6))
    print("C4 - TOTAL Item 1:", float(c4_T1 + c4_F1 + c4_L1))
    print("C4 - TOTAL Item 2:", float(c4_T2 + c4_F2 + c4_L2))
    print("C4 - TOTAL Item 3:", float(c4_T3 + c4_F3 + c4_L3))
    print("C4 - TOTAL Item 4:", float(c4_T4 + c4_F4 + c4_L4))
    print("C4 - TOTAL Item 5:", float(c4_T5 + c4_F5 + c4_L5))
    print("C4 - TOTAL Item 6:", float(c4_T6 + c4_F6 + c4_L6))
    print("C4 - TOTAL Classe:", float(c4_T1 + c4_T2 + c4_T3 + c4_T4 + c4_T5 + c4_T6 + c4_F1 + c4_F2 + c4_F3 + c4_F4 + c4_F5 + c4_F6 + c4_L1 + c4_L2 + c4_L3 + c4_L4 + c4_L5 + c4_L6))
    print('')

    print("RECEBIDO C1: ", float(((c1_T1 + c1_F1 + c1_L1) * 28) + ((c1_T2 + c1_F2 + c1_L2) * 30)))
    print("RECEBIDO C2: ", float(((c2_T1 + c2_F1 + c2_L1) * 7) + ((c2_T2 + c2_F2 + c2_L2) * 26) + ((c2_T3 + c2_F3 + c2_L3) * 30) + ((c2_T4 + c2_F4 + c2_L4) * 27)))
    print("RECEBIDO C3: ", float(((c3_T1 + c3_F1 + c3_L1) * 8) + ((c3_T2 + c3_F2 + c3_L2) * 29) + ((c3_T3 + c3_F3 + c3_L3) * 10) + ((c3_T4 + c3_F4 + c3_L4) * 35)))
    print("RECEBIDO C4: ", float(((c4_T1 + c4_F1 + c4_L1) * 23) + ((c4_T2 + c4_F2 + c4_L2) * 12) + ((c4_T3 + c4_F3 + c4_L3) * 21) + ((c4_T4 + c4_F4 + c4_L4) * 9) +
                                 ((c4_T5 + c4_F5 + c4_L5) * 26) + ((c4_T6 + c4_F6 + c4_L6) * 33)))

    print("Lucro Total: ", float(((c1_T1 + c1_F1 + c1_L1) * 28) + ((c1_T2 + c1_F2 + c1_L2) * 30) +
                                 ((c2_T1 + c2_F1 + c2_L1) * 7) + ((c2_T2 + c2_F2 + c2_L2) * 26) + ((c2_T3 + c2_F3 + c2_L3) * 30) + ((c2_T4 + c2_F4 + c2_L4) * 27) +
                                 ((c3_T1 + c3_F1 + c3_L1) * 8) + ((c3_T2 + c3_F2 + c3_L2) * 29) + ((c3_T3 + c3_F3 + c3_L3) * 10) + ((c3_T4 + c3_F4 + c3_L4) * 35) +
                                 ((c4_T1 + c4_F1 + c4_L1) * 23) + ((c4_T2 + c4_F2 + c4_L2) * 12) + ((c4_T3 + c4_F3 + c4_L3) * 21) + ((c4_T4 + c4_F4 + c4_L4) * 9) +
                                 ((c4_T5 + c4_F5 + c4_L5) * 26) + ((c4_T6 + c4_F6 + c4_L6) * 33)))

def main():
    rand = Random()
    rand.seed(int(time()))

    ea = ec.GA(rand)
    ea.selector = ec.selectors.tournament_selection
    ea.variator = [ec.variators.uniform_crossover, ec.variators.gaussian_mutation]
    ea.replacer = ec.replacers.steady_state_replacement

    ea.terminator = terminators.generation_termination

    ea.observer = [ec.observers.stats_observer, ec.observers.file_observer]

    final_pop = ea.evolve(generator=gera_populacao,
                          evaluator=avaliacao,
                          pop_size=1000,
                          maximize=True,
                          bounder=ec.Bounder(0, 456),
                          max_generations= 5000,
                          num_inputs=48,
                          crossover_points=1,
                          mutation_rate=0.25,
                          num_elites=1,
                          num_selected=48,
                          tournament_size=48)
                          #statistics_file=open('MOCHILA_stats.csv', 'w'),
                          #individuals_file=open('cargas_individuais.csv', 'w'))

    final_pop.sort(reverse=True)
    print(final_pop[0])

    calcula_fitness(final_pop[0].candidate[0], final_pop[0].candidate[1], final_pop[0].candidate[2], final_pop[0].candidate[3], final_pop[0].candidate[4], final_pop[0].candidate[5], final_pop[0].candidate[6], final_pop[0].candidate[7], final_pop[0].candidate[8], final_pop[0].candidate[9], final_pop[0].candidate[10], final_pop[0].candidate[11],
                    final_pop[0].candidate[12], final_pop[0].candidate[13], final_pop[0].candidate[14], final_pop[0].candidate[15], final_pop[0].candidate[16], final_pop[0].candidate[17], final_pop[0].candidate[18], final_pop[0].candidate[19], final_pop[0].candidate[20], final_pop[0].candidate[21], final_pop[0].candidate[22],
                    final_pop[0].candidate[23], final_pop[0].candidate[24], final_pop[0].candidate[25], final_pop[0].candidate[26], final_pop[0].candidate[27], final_pop[0].candidate[28], final_pop[0].candidate[29], final_pop[0].candidate[30], final_pop[0].candidate[31], final_pop[0].candidate[32], final_pop[0].candidate[33], final_pop[0].candidate[34],
                    final_pop[0].candidate[35], final_pop[0].candidate[36], final_pop[0].candidate[37], final_pop[0].candidate[38], final_pop[0].candidate[39], final_pop[0].candidate[40], final_pop[0].candidate[41], final_pop[0].candidate[42], final_pop[0].candidate[43], final_pop[0].candidate[44], final_pop[0].candidate[45],
                    final_pop[0].candidate[46], final_pop[0].candidate[47])

    calculo_solucao(final_pop[0].candidate[0], final_pop[0].candidate[1], final_pop[0].candidate[2], final_pop[0].candidate[3], final_pop[0].candidate[4], final_pop[0].candidate[5], final_pop[0].candidate[6], final_pop[0].candidate[7], final_pop[0].candidate[8], final_pop[0].candidate[9], final_pop[0].candidate[10], final_pop[0].candidate[11],
                    final_pop[0].candidate[12], final_pop[0].candidate[13], final_pop[0].candidate[14], final_pop[0].candidate[15], final_pop[0].candidate[16], final_pop[0].candidate[17], final_pop[0].candidate[18], final_pop[0].candidate[19], final_pop[0].candidate[20], final_pop[0].candidate[21], final_pop[0].candidate[22],
                    final_pop[0].candidate[23], final_pop[0].candidate[24], final_pop[0].candidate[25], final_pop[0].candidate[26], final_pop[0].candidate[27], final_pop[0].candidate[28], final_pop[0].candidate[29], final_pop[0].candidate[30], final_pop[0].candidate[31], final_pop[0].candidate[32], final_pop[0].candidate[33], final_pop[0].candidate[34],
                    final_pop[0].candidate[35], final_pop[0].candidate[36], final_pop[0].candidate[37], final_pop[0].candidate[38], final_pop[0].candidate[39], final_pop[0].candidate[40], final_pop[0].candidate[41], final_pop[0].candidate[42], final_pop[0].candidate[43], final_pop[0].candidate[44], final_pop[0].candidate[45],
                    final_pop[0].candidate[46], final_pop[0].candidate[47])

main()
