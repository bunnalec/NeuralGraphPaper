import torch

class CartPole:
    def __init__(self, device="cpu"):
        self.m_c = 1.0
        self.m_p = 0.1
        self.g = 9.8
        self.l = 0.5
        self.F_mag = 10
        self.mu_c = 0 # Set 0 cart friction
        self.mu_p = 0 # Set 0 pole friction
        self.dt = 0.02
        self.device = device

    def initialize(self, batch_size):
        self.theta = (torch.rand(batch_size).to(self.device) - .5) / 10
        self.omega = torch.zeros(batch_size).to(self.device)
        self.alpha = torch.zeros(batch_size).to(self.device)

        self.x = torch.zeros(batch_size).to(self.device)
        self.v = torch.zeros(batch_size).to(self.device)
        self.a = torch.zeros(batch_size).to(self.device)

        self.done = torch.zeros(batch_size).to(self.device)
        return self.get_state()

    def reset(self, indices):
        self.done[indices] = 0
        self.theta[indices] = (torch.rand(len(indices)).to(self.device) - .5) / 10
        self.omega[indices] = torch.zeros(len(indices)).to(self.device)
        self.alpha[indices] = torch.zeros(len(indices)).to(self.device)

        self.x[indices] = torch.zeros(len(indices)).to(self.device)
        self.v[indices] = torch.zeros(len(indices)).to(self.device)
        self.a[indices] = torch.zeros(len(indices)).to(self.device)

    def step(self, action):
        # Could have 2 or 3 actions depending on whether you want a 0 action
        F = torch.where(action == 0, -self.F_mag, 0) + torch.where(action == 1, self.F_mag, 0)
        
        # Continuous version
        # F = torch.clamp(action, -self.F_mag, self.F_mag)

        N_c = (self.m_c + self.m_p) * self.g - self.m_p * self.l * (self.alpha * torch.sin(self.theta) + torch.square(self.omega) * torch.cos(self.theta))
        
        nom = self.g * torch.sin(self.theta) + torch.cos(self.theta) * \
            ((-F-self.m_p * self.l * torch.square(self.omega) * (torch.sin(self.theta) + self.mu_c * torch.sign(N_c * self.v) * torch.cos(self.theta))) / (self.m_c + self.m_p) + self.mu_c * self.g * torch.sign(N_c * self.v)) \
            - self.mu_p * self.omega/(self.m_p * self.l)
        denom = self.l * (4/3 - (self.m_p * torch.cos(self.theta) / (self.m_c + self.m_p)) * (torch.cos(self.theta) - self.mu_c * torch.sign(N_c * self.v)))
        
        self.alpha = nom / denom

        self.a = (F + self.m_p * self.l * (torch.square(self.omega) * torch.sin(self.theta) - self.alpha * torch.cos(self.theta)) - self.mu_c * N_c * torch.sign(N_c * self.v)) / (self.m_c + self.m_p)

        # Only update ones not done
        self.v += self.a * self.dt * (1-self.done)
        self.x += self.v * self.dt * (1-self.done)
        self.omega += self.alpha * self.dt * (1-self.done)
        self.theta += self.omega * self.dt * (1-self.done)

        # Only reward ones not done
        # This is standard cartpole
        # reward = (1-self.done)

        # Update which ones are done
        # If it's not done and theta or x is out of range then mark as done
        theta_fail = torch.where(self.theta < -.2, 1, 0) + torch.where(self.theta > .2, 1, 0) # These are mut-ex
        x_fail = torch.where(self.x < -2, 1, 0) + torch.where(self.x > 2, 1, 0)
        self.done += (1-self.done) * (theta_fail + x_fail - theta_fail * x_fail)

        # I will do something a bit different.
        # -1 everytime you fail and never "done" per say
        reward = -self.done * 50 # Multiplying by 50 so approx. reward per step for doing nothing is -1
        if torch.any(self.done):
            self.reset(torch.argwhere(self.done == 1)[:, 0])

        # In fact just gonna give reward since I'm doing the reseting now
        return reward# , self.done # Not gonna return state so user can reset done states and call get_states to get those init states

    def get_state(self):
        return torch.stack([self.x, self.v, self.theta, self.omega], axis=1)