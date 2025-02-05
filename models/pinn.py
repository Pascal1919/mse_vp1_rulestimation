import torch
import torch.nn as nn

# x-NN Module (Self-Attention based feature extractor)
class xNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(xNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.features = input_dim
        self.multihead_attn = nn.MultiheadAttention(self.features, 1)  # Self-Attention layer
        self.Dense1 = nn.Linear(self.features, self.features)
        self.Dense2 = nn.Linear(self.features, self.hidden_dim)
        self.LN = nn.LayerNorm(self.features)
        self.activation = nn.ReLU()

    def forward(self, X):
        x, _ = self.multihead_attn(X, X, X)
        x = self.LN(x + X)
        x1 = self.Dense1(x)
        x1 = self.activation(x1 + x)
        return self.Dense2(x1)

# deepHPM Module (Self-Attention based physics model)
class DeepHPM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepHPM, self).__init__()
        self.hidden_dim = hidden_dim
        self.features = input_dim
        self.multihead_attn = nn.MultiheadAttention(self.features, 1)  # Self-Attention layer
        self.Dense1 = nn.Linear(self.features, self.features)
        self.Dense2 = nn.Linear(self.features, self.hidden_dim)
        self.LN = nn.LayerNorm(self.features)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        attn_out, _ = self.multihead_attn(x, x, x)
        x = self.LN(attn_out + x)
        x1 = self.Dense1(attn_out)
        x1 = self.activation(x1 + x)
        return self.Dense2(x1)


# MLP Module for mapping hidden states to RUL predictions
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.features = input_dim
        params = torch.ones(6)
        params = torch.full_like(params, 10, requires_grad=True)
        self.params = nn.Parameter(params)
        self.dnn = nn.Sequential(
            nn.Linear(self.features, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 6)
        )

    def forward(self, X):
        x = self.dnn(X)
        x = x * self.params
        return x.sum(dim=1)

# PINN Module
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, derivatives_order):
        super(PINN, self).__init__()
        self.hidden_dim = hidden_dim
        self.order = derivatives_order
        self.input_dim = 1 + self.hidden_dim * (self.order + 1)

        # Network Components
        self.xnn = xNN(input_dim, self.hidden_dim)
        self.mlp = MLP(self.hidden_dim + 1)
        self.deepHPM = DeepHPM(self.input_dim, 1)

    def forward(self, x, t):
        # data forward
        t.requires_grad_(True)
        hidden = self.xnn(x)
        hidden.requires_grad_(True)
        rul = self.mlp(torch.cat([hidden, t], dim=1))
        # physics informed
        rul = rul.reshape(-1, 1)
        rul.requires_grad_(True)
        rul_t = torch.autograd.grad(rul, t, grad_outputs=torch.ones_like(rul, requires_grad=True), retain_graph=True, create_graph=True, allow_unused=True)[0]
        rul_h = [rul]
        for _ in range(self.order):
            rul_ = torch.autograd.grad(rul_h[-1], hidden, grad_outputs=torch.ones_like(rul_h[-1], requires_grad=True), retain_graph=True, create_graph=True, allow_unused=True)[0]
            rul_h.append(rul_)
        deri = hidden
        for data in rul_h:
            deri = torch.cat([deri, data], dim=1)
        hpm = self.deepHPM(deri)
        f = rul_t - hpm
        return rul, f

# Physics-Based Loss Module
class PhysicsBasedLoss(nn.Module):
    def __init__(self, loss_lambda=100):
        super(PhysicsBasedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.loss_lambda = loss_lambda

    def forward(self, u, f, y):
        data_loss = torch.sqrt(self.mse(u, y))
        physics_loss = torch.sqrt(self.mse(f, torch.zeros_like(f).to(f.device))) * self.loss_lambda
        loss = data_loss + physics_loss
        return loss, data_loss, physics_loss
    

class AdaptivePhysicsBasedLoss(nn.Module):
    def __init__(
        self,
        alpha=0.5,
        l0=(1.0, 1.0),
        l1=(1.0, 1.0),
        lam=(1.0, 1.0),
        T=0.1,
        rho=0,
        loss_lambda=100
    ):
        """
        A single class that implements:
        (1) A data/physics loss computation
        (2) The relobralo weighting mechanism

        :param alpha: Interpolation factor for weighting strategy.
        :param l0, l1: Arrays/tuples of same length as (loss_u, loss_f)
                       controlling reference scales for each loss.
        :param lam: Base weighting factors for each loss.
        :param T: Temperature-like factor for softmax scaling.
        :param rho: Additional parameter to tweak weighting (0 <= rho <= 1).
        :param loss_lambda: Default multiplier for the physics-loss.
        """
        super(AdaptivePhysicsBasedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.l0 = l0
        self.l1 = l1
        self.lam = lam
        self.T = T
        self.rho = rho
        self.loss_lambda = loss_lambda

    def relobralo(self, loss_u, loss_f):
        """
        Adaptive balancing of two loss terms.
        Returns two scaling coefficients, one per loss term.
        """
        losses = [loss_u, loss_f]
        length = len(losses)
        length_tensor = torch.tensor(length, dtype=torch.float32, device=loss_u.device)

        # softmax( loss[i] / (l1[i]*T + eps) )
        temp1 = torch.softmax(
            torch.tensor([
                losses[i] / (self.l1[i] * self.T + 1e-12)
                for i in range(length)
            ], device=loss_u.device),
            dim=-1
        )
        # softmax( loss[i] / (l0[i]*T + eps) )
        temp2 = torch.softmax(
            torch.tensor([
                losses[i] / (self.l0[i] * self.T + 1e-12)
                for i in range(length)
            ], device=loss_u.device),
            dim=-1
        )

        # Weighted sums
        lambs_hat = temp1 * length_tensor     # from l1
        lambs0_hat = temp2 * length_tensor    # from l0

        # Combine them using alpha and rho
        lambs = []
        for i in range(length):
            lamb = (self.rho * self.alpha * self.lam[i]
                    + (1 - self.rho) * self.alpha * lambs0_hat[i]
                    + (1 - self.alpha) * lambs_hat[i])
            lambs.append(lamb)
        return lambs

    def forward(self, u, f, y):
        """
        :param u: Model output for data loss comparison
        :param f: Physics residual
        :param y: Ground-truth data
        :return: (final_loss, data_loss, physics_loss)
        """
        # 1. Compute raw losses
        data_loss = torch.sqrt(self.mse(u, y))
        physics_loss_raw = torch.sqrt(self.mse(f, torch.zeros_like(f).to(f.device)))

        # 2. Compute adaptive lambdas via relobralo
        lambs = self.relobralo(loss_u=data_loss, loss_f=physics_loss_raw)
        # lambs = [lambda_u, lambda_f]

        # 3. Combine
        # Note: We preserve the original idea of multiplying the physics term by loss_lambda
        loss = lambs[0] * data_loss + lambs[1] * (physics_loss_raw * self.loss_lambda)

        return loss, data_loss, physics_loss_raw

class Adan(torch.optim.Optimizer):
    """
    Implements the Adan optimization algorithm.

    Reference:
    Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models
    """
    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-8,
                 weight_decay=0.0, max_grad_norm=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        super(Adan, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adan does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq, exp_avg_diff, prev_grad = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_diff'], state['prev_grad']

                beta1, beta2, beta3 = group['betas']
                state['step'] += 1

                # Gradient clipping
                if group['max_grad_norm'] > 0:
                    grad_norm = grad.norm()
                    clip_coef = group['max_grad_norm'] / (grad_norm + 1e-6)
                    if clip_coef < 1:
                        grad = grad * clip_coef

                # Update exponential moving averages
                diff = grad - prev_grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_diff.mul_(beta2).add_(diff, alpha=1 - beta2)
                update = grad + beta2 * diff
                exp_avg_sq.mul_(beta3).addcmul_(update, update, value=1 - beta3)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr'] / denom

                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Parameter update
                p.data.addcdiv_(exp_avg + beta2 * exp_avg_diff, denom, value=-group['lr'])

                # Save current gradient
                prev_grad.copy_(grad)

        return loss
    
class Score(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_score(pred, true):
        score = 0
        for i in range(pred.shape[0]):
            h = pred[i] - true[i]
            if h >= 0:
                s = torch.exp(h / 10) - 1
            else:
                s = torch.exp(-h / 13) - 1
            score += s
        return score
