import numpy as np
from scipy.integrate import odeint
import sympy as sy
from scipy.linalg import expm


class RobotModel:
    def __init__(self, Ts=0.01, state=None):
        if state is not None:
            self.state = state
        else:
            self.state = np.array([0, 0, 0, 0, 0])
        self.Ts = Ts
        self.tauv = 1
        self.tauw = 1
        self.Kw = 1
        self.Kv = 1

        (self.A, self.B, self.C, self.D) = self.get_jacobians()
        x, y, theta, v, w = sy.symbols('x, y, theta, v, w')
        vcmd, wcmd = sy.symbols('vcmd, wcmd')
        self._A_func = sy.lambdify((x, y, theta, v, w, vcmd, wcmd), self.A, 'numpy')
        self._B_func = sy.lambdify((x, y, theta, v, w, vcmd, wcmd), self.B,'numpy')
        self.C = np.array(self.C, dtype=float)
        self.D = np.array(self.D, dtype=float)

    def yaw_gain(self, w_cmd):
        return self.Kw

    def dx_func(self, x, t, u):
        x, y, theta, v, w = x
        vcmd, wcmd = u
        dx = v*np.cos(theta)
        dy = v*np.sin(theta)
        dtheta = w
        dv = 1 / self.tauv * (self.Kv * vcmd - v)
        dw = (self.yaw_gain(wcmd) * wcmd - w) / self.tauw
        return [dx, dy, dtheta, dv, dw]

    def step(self, u):
        sol = odeint(self.dx_func, self.state, [0, self.Ts], args=(u,))
        self.state  = np.array(sol[-1, :])
        return self.state

    def f(self, state, u): # Useful for your MPC
        x, y, theta, v, w = np.asarray(state).squeeze()
        vcmd, wcmd = np.asarray(u).squeeze()
        next_state = np.array([
            x + self.Ts * v * np.cos(theta),
            y + self.Ts * v * np.sin(theta),
            theta + self.Ts * w,
            v + self.Ts / self.tauv * (self.Kv * vcmd - v),
            w + self.Ts / self.tauw *  (self.yaw_gain(wcmd) * wcmd - w)])
        return next_state.reshape(-1, 1)


    def discrete_step(self, u):
        x, y, theta, v, w = self.state
        vcmd, wcmd = u
        x_next = x + self.Ts * v * np.cos(theta)
        y_next = y + self.Ts * v * np.sin(theta)
        theta_next = theta + self.Ts * w
        v_next = v + self.Ts / self.tauv * (self.Kv * vcmd - v)
        w_next = w + self.Ts / self.tauw *  (self.yaw_gain(wcmd) * wcmd - w)
        self.state = np.array([x_next, y_next, theta_next, v_next, w_next])
        return self.state

    def get_jacobians(self):
        x, y, theta, v, w = sy.symbols('x y theta v w')
        vcmd, wcmd = sy.symbols('vcmd wcmd', real=True)

        x_vec = sy.Matrix([x, y, theta, v, w])
        u_vec = sy.Matrix([vcmd, wcmd])
        K = self.Kw

        # Discrete-time dynamics
        f_d = sy.Matrix([x + self.Ts * v * sy.cos(theta),
            y + self.Ts * v * sy.sin(theta),
            theta + self.Ts * w,
            v + self.Ts / self.tauv * (self.Kv * vcmd - v),
            w + self.Ts / self.tauw * (K * wcmd - w)])

        # Measurement model:
        h = sy.Matrix([x, y, theta, v, w])
        A = f_d.jacobian(x_vec)
        B = f_d.jacobian(u_vec)

        C = h.jacobian(x_vec)
        D = h.jacobian(u_vec)
        return A, B, C, D


    def evaluate_jacobians(self, state_est, u_k):
        x, y, theta, v, w = np.asarray(state_est.squeeze())
        vcmd, wcmd = np.asarray(u_k).squeeze()

        A = np.asarray(self._A_func(x, y, theta, v, w, vcmd, wcmd),dtype=float)
        B = np.asarray(self._B_func(x, y, theta,v, w,vcmd, wcmd),dtype=float)
        return A, B



if __name__ == "__main__":
    model = RobotModel()

    for i in range(10):
        x = model.step(u=[1, 0.5])
        print(x)
