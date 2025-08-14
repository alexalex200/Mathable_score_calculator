import cv2 as cv
import numpy as np
import os

height_board, width_board = 2800, 2800
square_size_h = 200
square_size_w = 200
pieces_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 27, 28, 30, 32,
                35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 80, 81, 90]

map = [['3', '', '', '', '', '', '3', '3', '', '', '', '', '', '3'],
       ['', '2', '', '', '/', '', '', '', '', '/', '', '', '2', ''],
       ['', '', '2', '', '', '-', '', '', '-', '', '', '2', '', ''],
       ['', '', '', '2', '', '', '+', '*', '', '', '2', '', '', ''],
       ['', '/', '', '', '2', '', '*', '+', '', '2', '', '', '/', ''],
       ['', '', '-', '', '', '', '', '', '', '', '', '-', '', ''],
       ['3', '', '', '*', '+', '', '', '', '', '*', '+', '', '', '3'],
       ['3', '', '', '+', '*', '', '', '', '', '+', '*', '', '', '3'],
       ['', '', '-', '', '', '', '', '', '', '', '', '-', '', ''],
       ['', '/', '', '', '2', '', '+', '*', '', '2', '', '', '/', ''],
       ['', '', '', '2', '', '', '*', '+', '', '', '2', '', '', ''],
       ['', '', '2', '', '', '-', '', '', '-', '', '', '2', '', ''],
       ['', '2', '', '', '/', '', '', '', '', '/', '', '', '2', ''],
       ['3', '', '', '', '', '', '3', '3', '', '', '', '', '', '3']]

game_matrix = []


def show_image(img, small=True):
    img = cv.resize(img, (0, 0), fx=0.2 if small else 1, fy=0.2 if small else 1)
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


def cut_board(img):
    img_blur = cv.GaussianBlur(img, (5, 5), 0)
    img_gray = cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY)

    hsv = cv.cvtColor(img_blur, cv.COLOR_BGR2HSV)
    l = np.array([14, 0, 0])
    u = np.array([255, 255, 255])
    table_mask = cv.inRange(hsv, l, u)
    img_table_removed = cv.bitwise_and(img_gray, img_gray, mask=table_mask)

    contours, _ = cv.findContours(img_table_removed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = None
    for contour in contours:
        if (cv.contourArea(contour) > max_area):
            max_area = cv.contourArea(contour)
            max_contour = contour

    epsilon = 0.02 * cv.arcLength(max_contour, True)
    corners = cv.approxPolyDP(max_contour, epsilon, True)
    destination = np.array([[height_board, 0], [0, 0], [0, width_board], [height_board, width_board]], dtype=np.float32)
    M = cv.getPerspectiveTransform(np.array(corners, dtype=np.float32), destination)
    board = cv.warpPerspective(img, M, (height_board, width_board))

    board = rotate_image(board, 0.4)

    remove_top = 370
    remove_bottom = 355
    remove_left = 360
    remove_right = 365

    cut_board = board[remove_top:height_board - remove_bottom, remove_left:width_board - remove_right]
    cut_board = cv.resize(cut_board, (width_board, height_board))
    #show_image(cut_board)
    return cut_board


def clear_piece_image(piece):
    global min_area
    piece = cv.cvtColor(piece, cv.COLOR_BGR2GRAY)
    piece = cv.threshold(piece, 127, 255, cv.THRESH_BINARY_INV)[1]
    piece = cv.GaussianBlur(piece, (3, 3), 0)

    edges = cv.Canny(piece, 50, 150, apertureSize=3)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    piece = cv.cvtColor(piece, cv.COLOR_GRAY2BGR)
    numbers = []
    for contour in contours:
        if cv.contourArea(contour) > 1500:
            numbers.append(contour)

    if (len(numbers) == 0):
        show_image(piece, False)
    for i in range(square_size_h):
        for j in range(square_size_w):
            if cv.pointPolygonTest(numbers[0], (j, i), False) == -1:
                if len(numbers) > 1:
                    if (cv.pointPolygonTest(numbers[1], (j, i), False) == -1):
                        piece[i, j] = 0
                else:
                    piece[i, j] = 0
    return piece


def cut_template_pieces():
    board1 = cv.imread('imagini_auxiliare/04.jpg')
    board2 = cv.imread('imagini_auxiliare/03.jpg')

    board1 = cut_board(board1)
    board2 = cut_board(board2)

    pieces1 = []
    for i in range(0, int(height_board / square_size_h), 2):
        for j in range(0, int(width_board / square_size_w), 2):
            if len(pieces1) >= 46:
                break
            pieces1.append(board1[i * square_size_h:(i + 1) * square_size_h, j * square_size_w:(j + 1) * square_size_w])

    pieces2 = []
    for i in range(5, 11):
        for j in range(4, int(width_board / square_size_w) - 2):
            if len(pieces2) >= 46:
                break
            pieces2.append(board2[i * square_size_h:(i + 1) * square_size_h, j * square_size_w:(j + 1) * square_size_w])

    for i in range(46):
        if not os.path.exists('piese/' + str(pieces_index[i])):
            os.makedirs('piese/' + str(pieces_index[i]))
        cv.imwrite('piese/' + str(pieces_index[i]) + '/' + str(pieces_index[i]) + '_0.jpg',
                   clear_piece_image(pieces1[i]))
        cv.imwrite('piese/' + str(pieces_index[i]) + '/' + str(pieces_index[i]) + '_1.jpg',
                   clear_piece_image(pieces2[i]))


def prep_board(board):
    board = cv.bilateralFilter(board, 9, 75, 75)
    board = cv.cvtColor(board, cv.COLOR_BGR2GRAY)
    T, board = cv.threshold(board, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return board


def cut_max_piece(prev_board, board):
    show_image(board)
    show_image(prev_board)
    diff = cv.absdiff(board, prev_board)
    diff = cv.morphologyEx(diff, cv.MORPH_OPEN, np.ones((11, 11), np.uint8))
    show_image(diff)
    max_mean = 0
    max_i, max_j = 0, 0
    for i in range(0, int(height_board / square_size_h)):
        for j in range(0, int(width_board / square_size_w)):
            if diff[i * square_size_h:(i + 1) * square_size_h,
               j * square_size_w:(j + 1) * square_size_w].mean() > max_mean:
                max_mean = diff[i * square_size_h:(i + 1) * square_size_h,
                           j * square_size_w:(j + 1) * square_size_w].mean()
                max_i, max_j = i, j
    piece = board[max_i * square_size_h:(max_i + 1) * square_size_h, max_j * square_size_w:(max_j + 1) * square_size_w]

    piece = clear_piece_image(piece)
    return piece, max_i, max_j

def template_match_piece(piece, templates=pieces_index):
    piece = cv.copyMakeBorder(piece, 50, 50, 50, 50, cv.BORDER_CONSTANT, value=[0, 0, 0])
    min_diff = 100000000
    min_diff_piece = None
    min_diff_piece_index = -1
    for i in range(len(templates)):
        for j in range(2):
            template = cv.imread('piese/' + str(templates[i]) + '/' + str(templates[i]) + '_' + str(j) + '.jpg')

            diff = cv.matchTemplate(piece, template, cv.TM_SQDIFF_NORMED)
            min_val, _, min_loc, _ = cv.minMaxLoc(diff)
            if min_val < min_diff:
                min_diff = min_val
                min_diff_piece = template
                min_diff_piece_index = templates[i]
    return min_diff_piece_index


def calc(i, j, n_i, n_j, nn_i, nn_j):
    pieces = []
    if (map[i][j] == '+' or map[i][j] in "23") and ((game_matrix[n_i, n_j] + game_matrix[nn_i, nn_j]) in pieces_index):
        pieces.append(game_matrix[n_i, n_j] + game_matrix[nn_i, nn_j])
    if (map[i][j] == '-' or map[i][j] in "23") and (
            (abs(game_matrix[n_i, n_j] - game_matrix[nn_i, nn_j])) in pieces_index):
        pieces.append(abs(game_matrix[n_i, n_j] - game_matrix[nn_i, nn_j]))
    if (map[i][j] == '*' or map[i][j] in "23") and ((game_matrix[n_i, n_j] * game_matrix[nn_i, nn_j]) in pieces_index):
        pieces.append(game_matrix[n_i, n_j] * game_matrix[nn_i, nn_j])
    if (map[i][j] == '/' or map[i][j] in "23"):
        if game_matrix[nn_i, nn_j] != 0 and game_matrix[n_i, n_j] % game_matrix[nn_i, nn_j] == 0 and (
                game_matrix[n_i, n_j] / game_matrix[nn_i, nn_j]) in pieces_index:
            pieces.append(np.int64(game_matrix[n_i, n_j] / game_matrix[nn_i, nn_j]))
        if game_matrix[n_i, n_j] != 0 and game_matrix[nn_i, nn_j] % game_matrix[n_i, n_j] == 0 and (
                game_matrix[nn_i, nn_j] / game_matrix[n_i, n_j]) in pieces_index:
            pieces.append(np.int64(game_matrix[nn_i, nn_j] / game_matrix[n_i, n_j]))
    pieces = list(set(pieces))
    return pieces


def posible_pieces(i, j):
    pieces = []

    if i >= 2 and game_matrix[i - 2, j] != -1 and game_matrix[i - 1, j] != -1:
        pieces.append(calc(i, j, i - 1, j, i - 2, j))
    if i <= 11 and game_matrix[i + 2, j] != -1 and game_matrix[i + 1, j] != -1:
        pieces.append(calc(i, j, i + 1, j, i + 2, j))
    if j >= 2 and game_matrix[i, j - 2] != -1 and game_matrix[i, j - 1] != -1:
        pieces.append(calc(i, j, i, j - 1, i, j - 2))
    if j <= 11 and game_matrix[i, j + 2] != -1 and game_matrix[i, j + 1] != -1:
        pieces.append(calc(i, j, i, j + 1, i, j + 2))
    pieces = [item for sublist in pieces for item in sublist]
    return pieces

#cut_template_pieces()

if not os.path.exists("462_Anca_Alexandru"):
    os.makedirs("462_Anca_Alexandru")

#Schimba path-ul catre folderul cu imaginile pe care se face testul
image_file_path = "testare/"

#modifca range-ul din for pentru a testa mai multe jocuri
for game in range(1, 5):
    game_matrix = np.array([[-1 for i in range(14)] for j in range(14)])
    game_matrix[6, 6] = 1
    game_matrix[6, 7] = 2
    game_matrix[7, 6] = 3
    game_matrix[7, 7] = 4
    turnes_file = open(image_file_path + str(game) + "_turns.txt", 'r')
    turnes = turnes_file.read().split('\n')
    scor_string = ""

    prev_board = cut_board(cv.imread('imagini_auxiliare/01.jpg'))
    prev_move = 0
    current_player = turnes[0].split(' ')[0]
    current_move = 0
    for turn in turnes[1:]:
        next_player = turn.split(' ')[0]
        next_move = int(turn.split(' ')[1]) - 1
        scor_string += current_player + ' ' + str(current_move + 1)
        turn_score = 0
        for move in range(current_move, next_move):
            image = cv.imread(image_file_path + str(game) + '_' + ('0' if move < 9 else '') + str(move + 1) + ".jpg")

            board = cut_board(image)
            piece, i, j = cut_max_piece(prev_board, board)
            templates = posible_pieces(i, j)
            if len(templates) == 0:
                templates = pieces_index
            if len(templates) == 1:
                index = templates[0]
            else:
                index = template_match_piece(piece, templates)
            game_matrix[i, j] = index
            with open("462_Anca_Alexandru/" + str(game) + '_' + ('0' if move < 9 else '') + str(move + 1) + ".txt",
                      'w') as f:
                f.write(str(i + 1) + chr(ord('A') + j) + ' ' + str(index))
            turn_score += templates.count(index) * index * (
                int(map[i][j]) if map[i][j] == '2' or map[i][j] == '3' else 1)
            prev_board = board
        scor_string += ' ' + str(turn_score) + '\n'
        current_player = next_player
        current_move = next_move

    turn_score = 0
    scor_string += current_player + ' ' + str(current_move + 1)
    for move in range(current_move, 50):
        image = cv.imread(image_file_path + str(game) + '_' + ('0' if move < 9 else '') + str(move + 1) + ".jpg")
        board = cut_board(image)
        piece, i, j = cut_max_piece(prev_board, board)
        templates = posible_pieces(i, j)
        if len(templates) == 0:
            templates = pieces_index
        if len(templates) == 1:
            index = templates[0]
        else:
            index = template_match_piece(piece, templates)
        game_matrix[i, j] = index
        with open("462_Anca_Alexandru/" + str(game) + '_' + ('0' if move < 9 else '') + str(move + 1) + ".txt",
                  'w') as f:
            f.write(str(i + 1) + chr(ord('A') + j) + ' ' + str(index))
        turn_score += index * (int(map[i][j]) if map[i][j] == '2' or map[i][j] == '3' else 1)
        prev_board = board

    scor_string += ' ' + str(turn_score) + '\n'
    with open("462_Anca_Alexandru/" + str(game) + "_scores.txt", 'w') as f:
        f.write(scor_string[:-1])
