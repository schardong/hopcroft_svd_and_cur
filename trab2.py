#!/usr/bin/python3
# -*- coding: utf-8 -*-


import numpy as np
from PIL import Image, ImageDraw


def exercicio_3_27(imagem_filepath):
    # Carrega a matriz da imagem em cada um dos canais de cores
    matriz = np.asarray(Image.open(imagem_filepath).convert('RGB'), dtype=float)
    matriz_red = matriz[:, :, 0]
    matriz_green = matriz[:, :, 1]
    matriz_blue = matriz[:, :, 2]

    # Guarda a norma de Frobenius de cada um dos canais para depois podermos ver quando da
    # norma nós mantivemos em cada uma das compressões.
    norma_frobenius_red = np.linalg.norm(matriz_red)
    norma_frobenius_green = np.linalg.norm(matriz_green)
    norma_frobenius_blue = np.linalg.norm(matriz_blue)
    print('norma de Frobenius do canal RED: {:.2f}'.format(norma_frobenius_red))
    print('norma de Frobenius do canal GREEN: {:.2f}'.format(norma_frobenius_green))
    print('norma de Frobenius do canal BLUE: {:.2f}'.format(norma_frobenius_blue))

    # Quantos valores singulares mantemos em cada uma das compressões
    fracao_valores_sing_a_manter = [0.05, 0.10, 0.25, 0.50]
    vetor_qtde_valores_singulares = [int(frac * matriz.shape[1])
                                     for frac in fracao_valores_sing_a_manter]

    # Decomposição SVD de cada um dos canais de cores
    u_red, d_red, v_red_transp = np.linalg.svd(matriz_red, full_matrices=False)
    u_green, d_green, v_green_transp = np.linalg.svd(matriz_green, full_matrices=False)
    u_blue, d_blue, v_blue_transp = np.linalg.svd(matriz_blue, full_matrices=False)

    for quantidade in vetor_qtde_valores_singulares:
        # Matrizes contendo os 'quantidade'-primeiros valores singulares na diagonal e zero nas
        # outras entradas
        zero_array = np.zeros(matriz.shape[1] - quantidade)
        s_red = np.diag(np.hstack((d_red[:quantidade], zero_array)))
        s_green = np.diag(np.hstack((d_green[:quantidade], zero_array)))
        s_blue = np.diag(np.hstack((d_blue[:quantidade], zero_array)))

        # As matrizes abaixo guardam a aproximação da imagem em cada um dos canais.
        matriz_aprox_red = np.dot(
            u_red, np.dot(s_red, v_red_transp)).astype(dtype=int)
        matriz_aprox_green = np.dot(
            u_green, np.dot(s_green, v_green_transp)).astype(dtype=int)
        matriz_aprox_blue = np.dot(
            u_blue, np.dot(s_blue, v_blue_transp)).astype(dtype=int)

        # Transformamos num array em que cada entrada possui os valores dos 3 canais de cores.
        matriz_aprox = np.array([list(zip(matriz_aprox_red[i, :],
                                          matriz_aprox_green[i, :],
                                          matriz_aprox_blue[i, :]))
                                 for i in range(matriz.shape[0])],
                                dtype=np.uint8)

        # Salva a imagem comprimida
        imagem_aprox = Image.fromarray(matriz_aprox, 'RGB')
        imagem_aprox.save('Lenna_{}.png'.format(quantidade))

        # Cálculo da porcentagem da norma de Frobenius mantida na imagem comprimida.
        soma_sigmas_quadrado_red = sum(sigma_i_red**2
                                       for sigma_i_red in d_red[:quantidade])
        soma_sigmas_quadrado_green = sum(sigma_i_green**2
                                         for sigma_i_green in d_green[:quantidade])
        soma_sigmas_quadrado_blue = sum(sigma_i_blue**2
                                        for sigma_i_blue in d_blue[:quantidade])
        print('-'*80)
        print('Quantidade de valores singulares mantidos: {}'.format(quantidade))
        print()
        print('Porcentagem mantida da norma de Frobenius RED: {:.2f}%'.format(
            100.0 * soma_sigmas_quadrado_red / (norma_frobenius_red**2)))
        print('Porcentagem mantida da norma de Frobenius GREEN: {:.2f}%'.format(
            100.0 * soma_sigmas_quadrado_green / (norma_frobenius_green**2)))
        print('Porcentagem mantida da norma de Frobenius BLUE: {:.2f}%'.format(
            100.0 * soma_sigmas_quadrado_blue / (norma_frobenius_blue**2)))


def exercicio_3_30(distance_matrix, vector_dimension_desired, is_debug=True):
    # Variáveis auxiliares para calcularmos (X * X.T) mais rapidamente
    n = distance_matrix.shape[0]
    assert n >= vector_dimension_desired
    sq_dist_matrix = np.square(distance_matrix)
    assert (sq_dist_matrix == sq_dist_matrix.T).all()
    normalized_sum_sq_dist_to_i = (1.0/n) * np.sum(sq_dist_matrix, axis=0)
    assert normalized_sum_sq_dist_to_i.shape[0] == n
    normalized_total_num_sq_dist = (1.0/n**2) * np.sum(sq_dist_matrix)
    assert isinstance(normalized_total_num_sq_dist, float)

    # Obtemos (X * X.T) pela fórmula do exercício 3.30a.
    x_times_x_transpose = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_times_x_transpose[i, j] = - 0.5 * (sq_dist_matrix[i, j]
                                                 - normalized_sum_sq_dist_to_i[i]
                                                 - normalized_sum_sq_dist_to_i[j]
                                                 + normalized_total_num_sq_dist)
    if is_debug:
        print('x_times_x_transpose')
        print(x_times_x_transpose)

    assert np.isclose(x_times_x_transpose, x_times_x_transpose.T).all()
    # Como X * X.T é simétrico, vamos obter X pela decomposição em autovalores e autovetores
    eigenvalues, eigenvectors = np.linalg.eigh(x_times_x_transpose)
    # Colunas de eigenvectors são os autovetores
    if is_debug:
        print('eigenvalues')
        print(eigenvalues)
        print()
        print('eigenvectors')
        print(eigenvectors)
        print()
        print('D')
        print(np.diag(eigenvalues))
        print()
        print('eigenvectors.T')
        print(eigenvectors.T)
        print()
        print('x_times_x_transpose')
        print(x_times_x_transpose)
        print()
        print('eigenvectors * D * eigenvectors.T')
        print(np.dot(eigenvectors, np.dot(np.diag(eigenvalues), eigenvectors.T)))
        print()
        # The assert below is not always satisfied because of rounding errors.
        # assert np.isclose(
        #     x_times_x_transpose,
        #     np.dot(eigenvectors, np.dot(np.diag(eigenvalues), eigenvectors.T))).all()
        print('sqrt(D)')
    max_eigenvalue = eigenvalues.max()
    sqrt_d = np.diag(np.sqrt(np.clip(eigenvalues, a_min=0.0, a_max=(max_eigenvalue + 1.0))))
    x = np.dot(eigenvectors, sqrt_d)
    if is_debug:
        print(sqrt_d)
        print()
        print('original:')
        print(x_times_x_transpose)
        print()
        print('obtained:')
        print(np.dot(x, x.T))
        print()
        # Tiramos o assert abaixo pq ele pode ser falso por causa de erros de aproximação, causados
        # pela função np.sqrt.
        # assert np.isclose(x_times_x_transpose, np.dot(x, x.T)).all()
        print('x obtido:')
        print(x)

    # O x obtido provavelmente terá zeros nas primeiras colunas, pois a decomposição em
    # autovalores retorna-os em ordem crescente. Portanto, ao reduzir a dimensão, queremos as
    # colunas mais à direita da matriz.
    return x[:, -vector_dimension_desired:]


def exercicio_3_31():
    dist_china = np.array(
        [[0, 125, 1239, 3026, 480, 3300, 3736, 1192, 2373, 1230, 979, 684],
         [125, 0, 1150, 1954, 604, 3330, 3740, 1316, 2389, 1207, 955, 661],
         [1239, 1150, 0, 1945, 1717, 3929, 4157, 2092, 1892, 2342, 2090, 1796],
         [3026, 1954, 1945, 0, 1847, 3202, 2457, 1570, 993, 3156, 2905, 2610],
         [480, 604, 1717, 1847, 0, 2825, 3260, 716, 2657, 1710, 1458, 1164],
         [3300, 3330, 3929, 3202, 2825, 0, 2668, 2111, 4279, 4531, 4279, 3985],
         [3736, 3740, 4157, 2457, 3260, 2668, 0, 2547, 3431, 4967, 4715, 4421],
         [1192, 1316, 2092, 1570, 716, 2111, 2547, 0, 2673, 2422, 2170, 1876],
         [2373, 2389, 1892, 993, 2657, 4279, 3431, 2673, 0, 3592, 3340, 3046],
         [1230, 1207, 2342, 3156, 1710, 4531, 4967, 2422, 3592, 0, 256, 546],
         [979, 955, 2090, 2905, 1458, 4279, 4715, 2170, 3340, 256, 0, 294],
         [684, 661, 1796, 2610, 1164, 3985, 4421, 1876, 3046, 546, 294, 0]])
    
    # Obtains cities coordinates
    coords_cidades = exercicio_3_30(dist_china, 2)

    print('=' * 80)
    print('Coordenadas das cidades do exercício 3.31:')
    print(coords_cidades) # Cada linha tem as coordenadas de uma cidade diferente.

    # Opens the image for drawing the points
    im = Image.open("China.png")
    draw = ImageDraw.Draw(im)

    # Scales the coordinates
    l = 0.064
    lambda_matrix = np.array([[l, 0],
                              [0, l]])
    coords_cidades = np.dot(coords_cidades, lambda_matrix)
    
    print('=' * 80)
    print("Coordenadas escaladas por uma matriz diagonal com lambda = %.3f:" % (l))
    print(coords_cidades)

    # Rotates the coordinates
    t = -0.75
    theta_matrix = np.array([[np.cos(t), -np.sin(t)],
                             [np.sin(t), np.cos(t)]])
    coords_cidades = np.dot(coords_cidades, theta_matrix)

    print('=' * 80)
    print("Coordenadas rotacionadas por uma matriz de rotação com theta = %.2f rad:" % (t))
    print(coords_cidades)

    # Translates the coordinates
    cst_x = 405
    cst_y = 245

    # Fills the centroid with yellow
    draw.rectangle([(cst_x - 1, cst_y - 1), (cst_x + 1, cst_y + 1)], 'yellow')
    
    coords_cidades += (cst_x, cst_y)
    
    print('=' * 80)
    print('Coordenadas transladadas pela constante de transladação (%d, %d):' % (cst_x, cst_y))
    print(coords_cidades)

    # Draws the coordinates
    for coords in coords_cidades:
        coords_int = (int(coords[0]), int(coords[1]))
        _, _, _, a = im.getpixel(coords_int)
        # If the pixel obtained from the image is not transparent, it means it's a valid
        # region in the map. Then it fills the pixel and the pixels around it with white.
        # Otherwise, it fills the pixel and the pixels around it with red. 
        if a == 255:
            draw.rectangle([(coords_int[0] - 1, coords_int[1] - 1), (coords_int[0] + 1, coords_int[1] + 1)], 'white')
        else:
            draw.rectangle([(coords_int[0] - 1, coords_int[1] - 1), (coords_int[0] + 1, coords_int[1] + 1)], 'red')
            print("Error: coordinate (%d, %d) not in China." % (coords[0], coords[1]))
    
    # Saves the image with the cities identified
    im.save("China_drawn.png", "PNG")


if __name__ == '__main__':
    ##exercicio_3_27('Lenna.png')
    exercicio_3_31()
