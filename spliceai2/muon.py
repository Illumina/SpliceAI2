import torch


class Muon(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            use_muon=True,
            lr=0.001,
            momentum=0.95,
            dampening=0.0,
            weight_decay=0.01,
            nesterov=True,
            ns_steps=5,
            eps=1e-8,
            adam_betas=(0.9, 0.999),
            maximize=False
    ):
        defaults = dict(
            use_muon=use_muon,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            adam_betas=adam_betas,
            maximize=maximize
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            use_muon = group["use_muon"]
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            beta1, beta2 = group["adam_betas"]
            maximize = group["maximize"]

            for param in params:
                grad = param.grad
                if grad is None:
                    continue
                grad = -grad if maximize else grad
                grad_shape = grad.shape

                if use_muon:
                    assert len(grad_shape) >= 2
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    grad = grad.add(buf, alpha=momentum)\
                        if nesterov else buf
                    grad_2d = grad.reshape(grad_shape[0], -1)
                    ortho_grad_2d = self._zeropower_via_newtonschulz5(
                        grad_2d, ns_steps, eps
                    )
                    ortho_grad = ortho_grad_2d.reshape(grad_shape)
                    step_size = lr * (
                        max(1, grad_2d.shape[0] / grad_2d.shape[1]) ** 0.5
                    )
                    param.mul_(1 - lr * weight_decay)
                    param.add_(ortho_grad, alpha=-step_size)
                else:
                    state = self.state[param]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(grad)
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad, grad, value=1 - beta2
                    )
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = lr / bias_correction1
                    denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                    param.mul_(1 - lr * weight_decay)
                    param.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

    @staticmethod
    def _zeropower_via_newtonschulz5(grad_2d, ns_steps, eps):
        grad_2d_shape = grad_2d.shape
        ns_transpose = grad_2d_shape[0] > grad_2d_shape[1]
        g = grad_2d.T if ns_transpose else grad_2d
        g = g / (g.norm() + eps)
        for _ in range(ns_steps):
            gg_t = g @ g.T
            g = 3.4445 * g + (-4.7750 * gg_t + 2.0315 * gg_t @ gg_t) @ g
        ortho_grad_2d = g.T if ns_transpose else g
        return ortho_grad_2d
