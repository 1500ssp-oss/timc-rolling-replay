
import numpy as np


def numerical_jacobian(func, x, eps=1e-5):
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(func(x), dtype=float)
    J = np.zeros((len(y0), len(x)))
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        J[:, i] = (np.asarray(func(xp)) - np.asarray(func(xm))) / (2 * eps)
    return J


class ExtendedKalmanFilter:
    def __init__(self, x0, P0, Q, R, f, h):
        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.f = f
        self.h = h

    def predict(self, u):
        F = numerical_jacobian(lambda xx: self.f(xx, u), self.x)
        self.x = self.f(self.x, u)
        self.P = F @ self.P @ F.T + self.Q
        return self.x

    def update(self, z):
        z = np.asarray(z, dtype=float)
        H = numerical_jacobian(lambda xx: self.h(xx), self.x)
        y = z - self.h(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        return self.x, y
