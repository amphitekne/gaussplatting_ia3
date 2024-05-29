import torch
from torch import nn

from gaussians.gaussian_model import GaussianModel
from gaussians.utils import build_rotation, inverse_sigmoid
from render import RasterOutputs


class Densifier:

    def __init__(self, gaussian: GaussianModel, optimizer: torch.optim.Optimizer, config: dict):
        self.gaussian = gaussian
        self.optimizer = optimizer
        self.config = config

        self.xyz_gradients = self.gaussian.xyz
        self.accumulated_iterations = torch.empty(0)
        self.max_radii = torch.empty(0)

        self._init_state()

    def _init_state(self):
        self.xyz_gradients = torch.zeros((self.gaussian.xyz.shape[0], 1), device="cuda")
        self.accumulated_iterations = torch.zeros((self.gaussian.xyz.shape[0], 1), device="cuda")
        self.max_radii = torch.zeros((self.gaussian.xyz.shape[0]), device="cuda")

    def _update_state(self, render_output: RasterOutputs):
        mask = render_output.visibility_mask
        self.xyz_gradients[mask] += torch.norm(render_output.screenspace_points.grad[mask, :2], dim=-1, keepdim=True)
        self.accumulated_iterations[mask] += 1
        self.max_radii[mask] = torch.max(self.max_radii[mask], render_output.radii[mask])

    def _add_parameters(self, parameters: dict):
        """
        Adds parameters to the gaussian and optimizer
        :param parameters:
        :return:
        """
        add_data = lambda x, y: torch.cat([x, y], dim=0)
        for group in self.optimizer.param_groups:
            param = group["params"][0]
            added_params = parameters[group["name"]]
            new_param = torch.nn.Parameter(
                add_data(param, added_params), True
            ).contiguous()
            state = self.optimizer.state.get(param, None)
            if state is not None:
                state["exp_avg"] = add_data(
                    state["exp_avg"], torch.zeros_like(added_params)
                )
                state["exp_avg_sq"] = add_data(
                    state["exp_avg_sq"], torch.zeros_like(added_params)
                )
                del self.optimizer.state[param]
                self.optimizer.state[new_param] = state
            group["params"][0] = new_param
            setattr(self.gaussian, group["name"], new_param)

    def _add_gaussians(self, parameters: dict):
        self._add_parameters(parameters)

        def reshape_tensor(tensor: torch.Tensor):
            if len(tensor.shape) == 2:
                new_tensor = torch.zeros((self.gaussian.xyz.shape[0], 1), device="cuda")
            else:
                new_tensor = torch.zeros((self.gaussian.xyz.shape[0]), device="cuda")
            new_tensor[:tensor.size(0)] = tensor
            return new_tensor

        self.xyz_gradients = reshape_tensor(self.xyz_gradients)
        self.accumulated_iterations = reshape_tensor(self.accumulated_iterations)
        self.max_radii = reshape_tensor(self.max_radii)

    def _prune_parameters(self, mask: torch.Tensor):
        for group in self.optimizer.param_groups:
            param = group["params"][0]
            new_param = torch.nn.Parameter(param[~mask], True).contiguous()
            state = self.optimizer.state.get(param, None)
            if state is not None:
                state["exp_avg"] = state["exp_avg"][~mask]
                state["exp_avg_sq"] = state["exp_avg_sq"][~mask]
                del self.optimizer.state[param]
                self.optimizer.state[new_param] = state
            group["params"][0] = new_param
            setattr(self.gaussian, group["name"], new_param)

    def _prune_gaussians(self, mask: torch.Tensor):
        self._prune_parameters(mask)

        self.xyz_gradients = self.xyz_gradients[~mask]
        self.accumulated_iterations = self.accumulated_iterations[~mask]
        self.max_radii = self.max_radii[~mask]

    def _clone(self, grads) -> int:
        grads = torch.norm(grads, dim=-1)
        big_grad_mask = torch.where(grads >= self.config["densify_grad_threshold"], True, False)
        under_reconstruction_mask = (torch.max(self.gaussian.get_size, dim=1).values
                                     <= self.config["densify_split_clone_threshold"] * self.gaussian.extent_radius)

        clone_mask = torch.logical_and(big_grad_mask, under_reconstruction_mask)

        parameters = {
            n: getattr(self.gaussian, n)[clone_mask]
            for n in self.gaussian.param_names
        }
        self._add_gaussians(parameters)
        return clone_mask.sum().item()

    def _split(self, grads):
        n_points_after_clone = self.gaussian.get_size.shape[0]
        new_grads = torch.zeros((n_points_after_clone), device="cuda")
        new_grads[:grads.shape[0]] = grads.squeeze()
        big_grad_mask = torch.where(new_grads >= self.config["densify_grad_threshold"], True, False)
        over_reconstruction_mask = (torch.max(self.gaussian.get_size, dim=1).values
                                    > self.config["densify_split_clone_threshold"] * self.gaussian.extent_radius)
        split_mask = torch.logical_and(big_grad_mask, over_reconstruction_mask)

        num_splits = 2
        scales = self.gaussian.get_size[split_mask].repeat(num_splits, 1)
        means = torch.zeros((scales.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=scales)
        rotations = build_rotation(self.gaussian.rotation[split_mask]).repeat(num_splits, 1, 1)
        new_xyz = torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1) + self.gaussian.xyz[split_mask].repeat(
            num_splits, 1)
        new_scaling = torch.log(self.gaussian.get_size[split_mask].repeat(num_splits, 1) / (0.8 * num_splits))

        parameters = {
            "xyz": new_xyz,
            "features_dc": self.gaussian.features_dc[split_mask].repeat(num_splits, 1, 1),
            "features_rest": self.gaussian.features_rest[split_mask].repeat(num_splits, 1, 1),
            "opacity": self.gaussian.opacity[split_mask].repeat(num_splits, 1),
            "scaling": new_scaling,
            "rotation": self.gaussian.rotation[split_mask].repeat(num_splits, 1),
        }
        self._add_gaussians(parameters)

        prune_mask = torch.cat((split_mask, torch.zeros(num_splits * split_mask.sum(), device="cuda", dtype=bool)))

        self._prune_gaussians(prune_mask)
        return prune_mask.sum().item()

    def _prune_big_points_and_opacity(self, prune_screen_big_points: bool, prune_world_big_points: bool):
        # Prune by opacity
        prune_mask = torch.where(
            self.gaussian.get_opacity.squeeze() < self.config["min_opacity_pruning"], True, False
        )
        n_opacity_prune = prune_mask.sum().item()

        big_points_vs = torch.zeros(self.max_radii.shape, device="cuda", dtype=bool)
        big_points_ws = torch.zeros(self.max_radii.shape, device="cuda", dtype=bool)

        if prune_screen_big_points:
            big_points_vs = self.max_radii > self.config["big_point_px_radius"]

        if prune_world_big_points:
            big_points_ws = self.gaussian.get_size.max(dim=1).values > self.config[
                "big_point_cov_extent"] * self.gaussian.extent_radius

        big_point_mask = torch.logical_or(big_points_vs, big_points_ws)
        n_big_point_prune = big_point_mask.sum().item()
        prune_mask = torch.logical_or(big_point_mask, prune_mask)

        self._prune_gaussians(prune_mask)

        return n_big_point_prune, n_opacity_prune

    def _reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.gaussian.get_opacity,
                      torch.ones_like(self.gaussian.get_opacity) * self.config["reset_opacity_value"]))
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "opacity":
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(opacities_new)
                stored_state["exp_avg_sq"] = torch.zeros_like(opacities_new)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(opacities_new.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        self.gaussian.opacity = optimizable_tensors["opacity"]

    def __call__(self, iteration: int, render_output: RasterOutputs):
        # Densification
        n_clone = 0
        n_split = 0
        n_prune_big = 0
        n_prune_opacity = 0

        if iteration < self.config["finish_densification"]:
            self._update_state(render_output)
            if iteration >= self.config["start_densification"] and iteration % self.config[
                "densification_interval"] == 0:

                grads = self.xyz_gradients / self.accumulated_iterations
                grads[grads.isnan()] = 0.0
                if len(self.gaussian) <= self.config["max_num_gaussians"]:
                    n_clone = self._clone(grads)
                    n_split = self._split(grads)
                prune_screen_big_points = False
                prune_world_big_points = True if iteration > self.config["reset_opacity_interval"] else False
                n_prune_big, n_prune_opacity = self._prune_big_points_and_opacity(prune_screen_big_points,
                                                                                  prune_world_big_points)

                if iteration % self.config["reset_opacity_interval"] == 0:
                    self._reset_opacity()
                self._init_state()
        return {
            "n_clone": n_clone,
            "n_split": n_split,
            "n_prune_big": n_prune_big,
            "n_prune_opacity": n_prune_opacity,
        }

    def capture(self):
        return (self.max_radii,
                self.xyz_gradients,
                self.accumulated_iterations,)

    def restore(self, model_args):
        (self.max_radii,
         self.xyz_gradients,
         self.accumulated_iterations,) = model_args
