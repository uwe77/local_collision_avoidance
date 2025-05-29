import numpy as np


class AdaptivePDWithBias:
    def __init__(self, kp=1.0, kd=0.1, kff=1.0, init_input_bias=0.0, 
                 output_limits=(-1.0, 1.0),
                 adapt_lr=0.01, bias_lr=0.001, gain_bounds=(0.1, 20)):
        self.kp = kp
        self.kd = kd
        self.kff = kff
        self.prev_error = 0.0
        self.input_bias = init_input_bias

        self.output_limits = output_limits
        self.adapt_lr = adapt_lr     # learning rate for gain update
        self.bias_lr = bias_lr       # learning rate for bias update
        self.gain_bounds = gain_bounds

    def compute(self, target, current, dt, adapt_mode=False):
        error = target*self.input_bias - current
        d_error = (error - self.prev_error) / dt
        self.prev_error = error

        # PD + feedforward + input bias
        output = (self.kp * error + 
                  self.kd * d_error + 
                  self.kff * target)

        output = np.clip(output, *self.output_limits)

        if adapt_mode:
            self.kp += self.adapt_lr * error * error
            self.kd += self.adapt_lr * d_error * error
            self.kff += self.adapt_lr * target * error

            # Clamp gains
            self.kp = np.clip(self.kp, *self.gain_bounds)
            self.kd = np.clip(self.kd, *self.gain_bounds)
            self.kff = np.clip(self.kff, *self.gain_bounds)

        return output

    def get_gains(self):
        return self.kp, self.kd, self.kff, self.input_bias
    

class ThrusterControllerPD:
    def __init__(self):
        self.linear_pd = AdaptivePDWithBias(kp=1.0, kd=0.1, kff=1.0, init_input_bias=5.47, output_limits=(0.0, 1.0))
        self.angular_pd = AdaptivePDWithBias(kp=10.0, kd=1.0, kff=1.0, init_input_bias=0.1)

    def step(self, cmd_vel, obs_vel, dt, adapt=True):
        linear_cmd, angular_cmd = cmd_vel
        linear_obs, angular_obs = obs_vel

        thrust = self.linear_pd.compute(linear_cmd, linear_obs, dt, adapt_mode=adapt)
        angle = self.angular_pd.compute(angular_cmd, angular_obs, dt, adapt_mode=adapt)

        # return np.array([thrust, angle])
        return cmd_vel

    def get_debug(self):
        return {
            'linear_gains': self.linear_pd.get_gains(),
            'angular_gains': self.angular_pd.get_gains()
        }