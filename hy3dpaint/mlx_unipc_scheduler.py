"""MLX UniPC updates for the Hunyuan paint scheduler configuration.

The implementation accepts the paint checkpoint's scaled-linear, trailing,
zero-terminal-SNR, v-prediction schedule.  It keeps latents and model-output
history as MLX arrays; scalar schedule coefficients stay in NumPy float32.
"""

import numpy as np
import mlx.core as mx


class MLXUniPCScheduler:
    def __init__(self, config):
        self.config = dict(config)
        required = {
            "beta_schedule": "scaled_linear",
            "prediction_type": "v_prediction",
            "timestep_spacing": "trailing",
            "rescale_betas_zero_snr": True,
        }
        for key, value in required.items():
            if self.config.get(key) != value:
                raise ValueError(f"unsupported UniPC {key}: {self.config.get(key)}")
        for key, value in {
            "solver_order": 2, "solver_type": "bh2", "predict_x0": True,
            "lower_order_final": True, "thresholding": False,
            "use_karras_sigmas": False, "final_sigmas_type": "zero",
        }.items():
            if self.config.get(key, value) != value:
                raise ValueError(f"unsupported UniPC {key}: {self.config.get(key)}")
        if self.config.get("disable_corrector", []):
            raise ValueError("unsupported UniPC disable_corrector")
        if self.config.get("solver_p") is not None:
            raise ValueError("unsupported UniPC solver_p")

        count = int(self.config["num_train_timesteps"])
        betas = np.linspace(
            np.sqrt(self.config["beta_start"]), np.sqrt(self.config["beta_end"]),
            count, dtype=np.float32,
        ) ** 2
        alphas_bar_sqrt = np.sqrt(np.cumprod(1.0 - betas, dtype=np.float32))
        first, last = alphas_bar_sqrt[0].copy(), alphas_bar_sqrt[-1].copy()
        alphas_bar_sqrt -= last
        alphas_bar_sqrt *= first / (first - last)
        alphas_bar = alphas_bar_sqrt ** 2
        alphas = np.concatenate([alphas_bar[:1], alphas_bar[1:] / alphas_bar[:-1]])
        alphas_cumprod = np.cumprod(alphas, dtype=np.float32)
        alphas_cumprod[-1] = 2 ** -24
        self._training_sigmas = np.sqrt((1.0 - alphas_cumprod) / alphas_cumprod)
        self.timesteps = None
        self.sigmas = None
        self.step_index = 0
        self.model_outputs = [None, None]
        self.last_sample = None
        self.this_order = 0
        self.lower_order_nums = 0

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def set_timesteps(self, count):
        if count < 1:
            raise ValueError("num_inference_steps must be positive")
        train_count = int(self.config["num_train_timesteps"])
        self.timesteps = np.arange(train_count, 0, -train_count / count).round().astype(np.int64) - 1
        selected = np.interp(self.timesteps, np.arange(train_count), self._training_sigmas)
        self.sigmas = np.concatenate([selected, [0.0]]).astype(np.float32)
        self.step_index = 0
        self.model_outputs = [None, None]
        self.last_sample = None
        self.this_order = 0
        self.lower_order_nums = 0

    def _coefficients(self, index):
        sigma = np.float32(self.sigmas[index])
        alpha = np.float32(1.0) / np.sqrt(sigma * sigma + np.float32(1.0))
        return alpha, sigma * alpha

    def _lambda(self, index):
        alpha, sigma = self._coefficients(index)
        if sigma == 0:
            return np.float32(np.inf)
        return np.log(alpha) - np.log(sigma)

    def _predictor(self, sample, order):
        index = self.step_index
        alpha_t, sigma_t = self._coefficients(index + 1)
        _, sigma_s0 = self._coefficients(index)
        h = self._lambda(index + 1) - self._lambda(index)
        h_phi_1 = np.expm1(-h)
        base = sample * (sigma_t / sigma_s0) - self.model_outputs[-1] * (alpha_t * h_phi_1)
        if order == 1:
            return base
        lambda_si = self._lambda(index - 1)
        rk = (lambda_si - self._lambda(index)) / h
        difference = (self.model_outputs[-2] - self.model_outputs[-1]) / rk
        return base - difference * (alpha_t * h_phi_1 * np.float32(0.5))

    def _corrector(self, model_output, last_sample, current_sample, order):
        index = self.step_index
        alpha_t, sigma_t = self._coefficients(index)
        _, sigma_s0 = self._coefficients(index - 1)
        h = self._lambda(index) - self._lambda(index - 1)
        hh = -h
        h_phi_1 = np.expm1(hh)
        base = last_sample * (sigma_t / sigma_s0) - self.model_outputs[-1] * (alpha_t * h_phi_1)
        delta = model_output - self.model_outputs[-1]
        if order == 1:
            return base - delta * (alpha_t * h_phi_1 * np.float32(0.5))

        lambda_si = self._lambda(index - 2)
        rk = (lambda_si - self._lambda(index - 1)) / h
        h_phi_k = h_phi_1 / hh - np.float32(1.0)
        b0 = h_phi_k / h_phi_1
        h_phi_k = h_phi_k / hh - np.float32(0.5)
        b1 = h_phi_k * np.float32(2.0) / h_phi_1
        rho0 = (b1 - b0) / (rk - np.float32(1.0))
        rho1 = b0 - rho0
        previous_delta = (self.model_outputs[-2] - self.model_outputs[-1]) / rk
        return base - (previous_delta * rho0 + delta * rho1) * (alpha_t * h_phi_1)

    def step(self, model_output, timestep, sample):
        if self.timesteps is None:
            raise ValueError("set_timesteps must be called before step")
        if self.step_index >= len(self.timesteps):
            raise ValueError("UniPC schedule is exhausted")
        expected = int(self.timesteps[self.step_index])
        if int(timestep) != expected:
            raise ValueError(f"unexpected timestep: {timestep}; expected {expected}")
        if model_output.shape != sample.shape or model_output.dtype != sample.dtype:
            raise ValueError("model output and sample must have the same shape and dtype")

        alpha_s, sigma_s = self._coefficients(self.step_index)
        converted = sample * alpha_s - model_output * sigma_s
        if self.step_index > 0 and self.last_sample is not None:
            sample = self._corrector(converted, self.last_sample, sample, self.this_order)

        self.model_outputs[0] = self.model_outputs[1]
        self.model_outputs[1] = converted
        remaining = len(self.timesteps) - self.step_index
        self.this_order = min(2, remaining, self.lower_order_nums + 1)
        self.last_sample = sample
        previous = self._predictor(sample, self.this_order).astype(sample.dtype)
        self.lower_order_nums = min(2, self.lower_order_nums + 1)
        self.step_index += 1
        return previous
