import numpy as np
import math

number_of_points = 10000


def function_calculator(x):
    y = math.sin(x * math.pi)
    return [x, y]


def main():
    point_holder = []
    fo = open("training_points.csv", "w")
    for i in range(0, number_of_points):
        x = np.random.uniform(low=0.0, high=1.0)
        current_point = function_calculator(x)
        fo.write("{},{}\n".format(current_point[0], current_point[1]))
    fo.close()
    fo = open("testing_points.csv", "w")
    for i in range(0, int(.2 * 1 / .8 * number_of_points)):
        x = np.random.uniform(low=0.0, high=1.0)
        current_point = function_calculator(x)
        print(current_point)
        fo.write("{},{}\n".format(current_point[0], current_point[1]))
    fo.close()


class point_make():
    def __init__(self, number_of_points):
        self.number_of_points = number_of_points

    def x_gen(self):
        return np.random.uniform(low=0.0, high=1.0, size=self.number_of_points)

    def y_gen(self, x_points):
        y_cor = []
        for x in x_points:
            y_cor.append(math.sin(x * 2 * math.pi))
        return y_cor

    def make_train_points(self, x_list, y_list):
        fo = open('train_points.csv', "w")
        for i in range(0, len(x_list)):
            fo.write("{},{}\n".format(x_list[i], y_list[i]))
        fo.close()

    def make_test_points(self, x_list, y_list):
        fo = open('test_points.csv', "w")
        for i in range(0, len(x_list)):
            fo.write("{},{}\n".format(x_list[i], y_list[i]))
        fo.close()


if __name__ == "__main__":
    points = point_make(number_of_points)
    train_points = points.x_gen()
    test_points = points.x_gen()
    points.make_train_points(test_points, points.y_gen(test_points))
    points.make_test_points(train_points, points.y_gen(train_points))














