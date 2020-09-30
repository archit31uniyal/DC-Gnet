import numpy as np
import cv2
from PIL import Image

def swap(a, b):
    return b, a

def ISNT(cup, disc, eye):
    ver_cup_len = 0
    ver_cup_dia= []

    for i in range(cup.shape[1]):
        for j in range(cup.shape[0]):
            if j == cup.shape[0]-1:
                ver_cup_dia.append(ver_cup_len)
                ver_cup_len = 0
            elif cup[j][i] == 255:
                ver_cup_len += 1

    # print(len(ver_cup_dia), '\n')

    ########################

    ver_cup_diameter = max(ver_cup_dia)
    # print(ver_cup_diameter, '\n')

    ########################

    x = 0
    y = 0
    for d in range(len(ver_cup_dia)):
        if ver_cup_dia[d] == ver_cup_diameter:
            x = d
            break

    for e in range(x, len(ver_cup_dia)):
        if ver_cup_dia[e] != ver_cup_diameter:
            y = e
            break

    # print((x, y), '\n')

    #########################

    ver_cup_dia_ind = x + round((y - x)/2)
    # print("Vertical :", ver_cup_dia_ind)



    ###############################################################################################

    hor_cup_len = 0
    hor_cup_dia= []

    for i in range(cup.shape[1]):
        for j in range(cup.shape[0]):
            if j == cup.shape[0]-1:
                hor_cup_dia.append(hor_cup_len)
                hor_cup_len = 0
            elif cup[i][j] == 255:
                hor_cup_len += 1

    # print(len(hor_cup_dia), '\n')

    ########################

    hor_cup_diameter = max(hor_cup_dia)
    # print(hor_cup_diameter, '\n')

    ########################

    g = 0
    h = 0
    for d in range(len(hor_cup_dia)):
        if hor_cup_dia[d] == hor_cup_diameter:
            g = d
            break

    for e in range(x, len(hor_cup_dia)):
        if hor_cup_dia[e] != hor_cup_diameter:
            h = e
            break

    # print((g, h), '\n')

    #########################

    hor_cup_dia_ind = g + round((h - g)/2)
    # print("Horizontal :", hor_cup_dia_ind)

    ###########################################################################################################

    disc_bound_s = 0
    for i in range(disc.shape[0]):
        if disc[i][ver_cup_dia_ind] == 0:
            disc_bound_s += 1
        elif disc[i][ver_cup_dia_ind] == 255:
            break

    # print(disc_bound_s)

    cup_bound_s = 0
    for i in range(cup.shape[0]):
        if cup[i][ver_cup_dia_ind] == 0:
            cup_bound_s += 1
        elif cup[i][ver_cup_dia_ind] == 255:
            break

    # print(cup_bound_s)

    isnt_s = cup_bound_s - disc_bound_s
    # print("S :", isnt_s)

    ##############################################################################################################

    i = list(range(disc.shape[0]))
    i.reverse()
    j = list(range(cup.shape[0]))
    j.reverse()

    disc_bound_i = 0
    cup_bound_i = 0

    for p in i:
        if disc[p][ver_cup_dia_ind] == 0:
            disc_bound_i += 1
        elif disc[p][ver_cup_dia_ind] == 255:
            break

    for q in j:
        if cup[q][ver_cup_dia_ind] == 0:
            cup_bound_i += 1
        elif cup[q][ver_cup_dia_ind] == 255:
            break

    # print(cup_bound_i, disc_bound_i, '\n')

    isnt_i = cup_bound_i - disc_bound_i
    # print("I : ", isnt_i)

    #############################################################################################################

    disc_bound_t = 0
    for i in range(disc.shape[1]):
        if disc[hor_cup_dia_ind][i] == 0:
            disc_bound_t += 1
        elif disc[hor_cup_dia_ind][i] == 255:
            break

#     print(disc_bound_s)

    cup_bound_t = 0
    for i in range(cup.shape[1]):
        if cup[hor_cup_dia_ind][i] == 0:
            cup_bound_t += 1
        elif cup[hor_cup_dia_ind][i] == 255:
            break

    # print(cup_bound_s)

    isnt_t = cup_bound_t - disc_bound_t
    # print("T :", isnt_t)

    #############################################################################################################

    i = list(range(disc.shape[1]))
    i.reverse()
    j = list(range(cup.shape[1]))
    j.reverse()

    disc_bound_n = 0
    cup_bound_n = 0

    for p in i:
        if disc[hor_cup_dia_ind][p] == 0:
            disc_bound_n += 1
        elif disc[hor_cup_dia_ind][p] == 255:
            break

    for q in j:
        if cup[hor_cup_dia_ind][q] == 0:
            cup_bound_n += 1
        elif cup[hor_cup_dia_ind][q] == 255:
            break

    isnt_n = cup_bound_n - disc_bound_n
    # print("N : ", isnt_n)

    #############################################################################################################

    cup_dias = [ver_cup_diameter, hor_cup_diameter]
    
    if eye == 'r':
        isnt_n, isnt_t = swap(isnt_n, isnt_t)

    return list([isnt_i, isnt_s, isnt_n, isnt_t]), cup_dias



def DDLS(cup_img, disc_img, precision_angle=10):
    
    rim_ind_line = []

    min_rim = []
    disc_diameter = []
    
    disc_dias = []
    
    ##############################################
    
    for angle in np.arange(0, 360, precision_angle):
        cup = cup_img.rotate(angle)
        disc = disc_img.rotate(angle)
        
        #################################
    
        cup = np.array(cup)
        disc = np.array(disc)
    
        #################################
        
        cup_width = []
        cup_boundary = 0
        for i in range(cup.shape[1]):
            for j in range(cup.shape[0]):
                if j == cup.shape[0]-1:
                    cup_width.append(cup_boundary)
                    cup_boundary = 0
                    break
                elif cup[j][i] == 255:
                    cup_width.append(cup_boundary)
                    cup_boundary = 0
                    break
                else:
                    cup_boundary += 1
        
        #################################
    
        n = 0
        m = 0
        for s in range(len(cup_width)):
            if cup_width[s] != 511:
                n = s
                break

        for t in range(n, len(cup_width)):
            if cup_width[t] == 511:
                m = t
                break
        
        #################################
    
        disc_width = []
        disc_boundary = 0

        for k in range(n, m):
            for l in range(disc.shape[0]):
                if l == disc.shape[0]-1:
                    disc_width.append(disc_boundary)
                    disc_boundary = 0
                    break
                elif disc[l][k] == 255:
                    disc_width.append(disc_boundary)
                    disc_boundary = 0
                    break
                else:
                    disc_boundary += 1        
        
        #################################
    
        cup_width = np.array(cup_width[n:m])
        disc_width = np.array(disc_width)
        
        #################################
    
        rim_width = cup_width - disc_width
        min_rim.append(min(rim_width))

        line_ind = np.where(rim_width == min(rim_width))[0]
        
        rim_ind_line.append(n + line_ind[len(line_ind)//2])

        #################################
        
        diameter = []
        dia = 0
        for p in range(disc.shape[1]):
            for q in range(disc.shape[0]):
                if q == disc.shape[0] - 1:
                    if dia > 0:
                        diameter.append(dia)
                    dia = 0
                    break
                if disc[q][p] == 255:
                    dia += 1
                    
        disc_diameter.append(max(diameter))
        
        # 0-vertical    # 1-hor
        
        if angle == 0 or angle == 90:
            disc_dias.append(max(diameter))
        
    ##############################################  

    print("Minimum Rim Width : {}".format(min(min_rim)))
    print("Disc Diameter : {}".format(max(disc_diameter)), '\n')
        
    return min(min_rim)/max(disc_diameter), disc_dias, min(min_rim), min_rim.index(min(min_rim))*precision_angle, rim_ind_line[min_rim.index(min(min_rim))]


