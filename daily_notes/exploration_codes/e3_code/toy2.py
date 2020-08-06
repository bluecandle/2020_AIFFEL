# from sklearn.metrics import mean_squared_error
# import numpy as np

# # y_true = [3, -0.5, 2, 7]
# # y_pred = [2.5, 0.0, 2, 8]
# # mean_squared_error(y_true, y_pred)

# y_true = [[0.5, 1], [-1, 1], [7, -6]]
# y_pred = [[0, 2], [-1, 2], [8, -5]]
# # mean_squared_error(y_true, y_pred)

# def error(y_true,y_pred) :
#     s = 0
#     n = len(y_true)
#     for i in range(n):
#         s += np.sqrt(y_true[i]-y_pred[i])
#     return s/n

# print(error(y_true,y_pred))
# print(mean_squared_error(y_true, y_pred))