"""Tactile sensor demo.

A hemispherical probe (mesh geom whose vertices act as taxels) falls onto a
block and is then pushed around by external forces. The MuJoCo tactile sensor
(mjSENS_TACTILE) reports per-taxel penetration depth, which this demo prints as
a compact summary.

Note: per-taxel magnitudes depend strongly on contact depth and solver
settings; with a stiff default contact, only the taxels nearest the contact
point register (small) values, and more taxels engage as the probe is
pressed deeper into the block.

Run with:
  uv run python scripts/demos/tactile_sensor.py
  uv run python scripts/demos/tactile_sensor.py --viewer viser         # Viser
  uv run python scripts/demos/tactile_sensor.py --timestep 0.005 --decimation 4
"""

from __future__ import annotations

import dataclasses
import os
from typing import Literal

import mujoco
import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sensor.builtin_sensor import ObjRef
from mjlab.sensor.tactile_sensor import TactileSensorCfg
from mjlab.sim.sim import MujocoCfg, SimulationCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

NUM_ENVS = 1

# External force profile applied to the probe body. The x force is a PD
# controller tracking a sinusoidal sweep target so the probe rocks in place.
SWEEP_X_AMPLITUDE = 0.35  # sweep target amplitude [m]
SWEEP_PERIOD_S = 6.0  # sweep period [s]
SWEEP_KP = 80.0  # PD proportional gain
SWEEP_KD = 20.0  # PD derivative gain
PRESS_FORCE_Z = -80.0  # extra downward press force


def create_probe_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.modelname = "tactile_probe"

  # Coarse sphere mesh; each vertex becomes one taxel (42 taxels for
  # subdivision level 1).
  mesh = spec.add_mesh()
  mesh.name = "probe_mesh"
  mesh.make_sphere(subdivision=1)

  # Fixed block for the probe to rest on (top face at z=1.0). The builtin
  # sphere mesh is unit-radius, so the probe rests with its center around
  # z ~ 2.0 once contacts engage.
  block = spec.worldbody.add_geom()
  block.name = "block_geom"
  block.type = mujoco.mjtGeom.mjGEOM_BOX
  block.size[:] = (1.5, 1.5, 0.15)
  block.pos[:] = (0, 0, 0.85)
  block.rgba[:] = (0.4, 0.5, 0.6, 1.0)

  # Probe: free-floating body carrying the sensor geom.
  probe = spec.worldbody.add_body()
  probe.name = "probe"
  probe.pos[:] = (0, 0, 2.3)

  joint = probe.add_joint()
  joint.name = "probe_joint"
  joint.type = mujoco.mjtJoint.mjJNT_FREE
  joint.damping[:] = 2.0

  geom = probe.add_geom()
  geom.name = "sensor_geom"
  geom.type = mujoco.mjtGeom.mjGEOM_MESH
  geom.meshname = "probe_mesh"
  geom.mass = 0.5  # light probe settles into the block surface
  geom.rgba[:] = (0.9, 0.4, 0.1, 1.0)

  return spec


def create_env_cfg(
  timestep: float,
  decimation: int,
  with_sensor: bool,
  debug_vis_scale: float = 0.3,
) -> ManagerBasedRlEnvCfg:
  probe_entity_cfg = EntityCfg(
    spec_fn=create_probe_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 2.3)),
  )

  sensors: tuple = ()
  if with_sensor:
    sensors = (
      TactileSensorCfg(
        name="tactile",
        obj=ObjRef(type="mesh", name="probe_mesh", entity="probe"),
        ref=ObjRef(type="geom", name="sensor_geom", entity="probe"),
        cutoff=0.0,
        debug_vis=True,
        debug_vis_scale=debug_vis_scale,
      ),
    )

  cfg = ManagerBasedRlEnvCfg(
    decimation=decimation,
    scene=SceneCfg(
      num_envs=NUM_ENVS,
      env_spacing=3.0,
      extent=2.0,
      entities={"probe": probe_entity_cfg},
      sensors=sensors,
    ),
    sim=SimulationCfg(
      njmax=50,
      mujoco=MujocoCfg(timestep=timestep),
    ),
  )

  cfg.viewer.body_name = "probe/probe"
  cfg.viewer.distance = 4.0
  cfg.viewer.elevation = -15.0
  cfg.viewer.azimuth = 90.0

  return cfg


def sweep_force(
  env: ManagerBasedRlEnv | RslRlVecEnvWrapper,
  body_id: int,
  dofadr: int,
  step: int,
  timestep: float,
  decimation: int,
) -> float:
  """PD force tracking a sinusoidal x target, so the probe rocks in place."""
  t = step * timestep * decimation
  x_target = SWEEP_X_AMPLITUDE * np.sin(t / SWEEP_PERIOD_S * 2 * np.pi)
  x = env.unwrapped.sim.data.xpos[0, body_id, 0].item()
  vx = env.unwrapped.sim.data.qvel[0, dofadr].item()
  return SWEEP_KP * (x_target - x) - SWEEP_KD * vx


class TactilePrintPolicy:
  """Zero-action policy that drives the probe and prints tactile readings.

  Taxel debug visualization is handled by the viewer via
  ``BuiltinSensor.debug_vis`` (toggle it with the "tactile" checkbox in the
  viser side panel)."""

  def __init__(
    self,
    env: ManagerBasedRlEnv | RslRlVecEnvWrapper,
    device: str,
    timestep: float,
    decimation: int,
    print_every: int,
  ) -> None:
    self.env = env
    self.device = device
    self.timestep = timestep
    self.decimation = decimation
    self.print_every = print_every
    self.step_count = 0

    self.sensor = env.unwrapped.scene["probe/tactile"]
    mjm = env.unwrapped.sim.mj_model
    self.mj_sensor = mjm.sensor("probe/tactile")
    self.num_taxels = int(mjm.mesh_vertnum[self.mj_sensor.objid[0]])
    self.body_id = mjm.body("probe/probe").id
    self.dofadr = int(mjm.body_dofadr[self.body_id])

    print(f"Tactile sensor: {self.num_taxels} taxels, dim={self.mj_sensor.dim[0]}")

  def __call__(self, obs) -> torch.Tensor:
    # Rock the probe sideways (PD sweep) and press it into the block.
    step = self.step_count
    fx = sweep_force(
      self.env, self.body_id, self.dofadr, step, self.timestep, self.decimation
    )
    fz = PRESS_FORCE_Z if step > 200 else 0.0
    self.env.unwrapped.sim.data.xfrc_applied[0, self.body_id, 0:3] = torch.tensor(
      [fx, 0.0, fz], device=self.device, dtype=torch.float32
    )

    self.step_count += 1

    if step % self.print_every == 0:
      # Raw sensordata is channel-major: [depth, slip_u, slip_v] blocks of
      # num_taxels each. Channel 0 is per-taxel penetration depth.
      readings = self.sensor.data[0].reshape(3, self.num_taxels)
      depths = readings[0]
      active = int((depths > 1e-6).sum())
      probe = self.env.unwrapped.sim.data.xpos[0, self.body_id].cpu().numpy()
      print(
        f"step {step:5d} | probe x={probe[0]:+.2f} z={probe[2]:.2f} | "
        f"active taxels: {active:2d}/{self.num_taxels} | "
        f"max depth: {depths.max().item():.3f}"
      )

    return torch.zeros(self.env.unwrapped.action_space.shape, device=self.device)


class ZeroPolicy:
  """Zero-action policy for the sensor-less A/B mode."""

  def __init__(
    self,
    env: ManagerBasedRlEnv | RslRlVecEnvWrapper,
    device: str,
    timestep: float = 0.002,
    decimation: int = 1,
  ) -> None:
    self.env = env
    self.device = device
    self.timestep = timestep
    self.decimation = decimation
    self.step_count = 0
    mjm = env.unwrapped.sim.mj_model
    self.body_id = mjm.body("probe/probe").id
    self.dofadr = int(mjm.body_dofadr[self.body_id])
    print("No tactile sensor in model; taxel output disabled.")

  def __call__(self, obs) -> torch.Tensor:
    step = self.step_count
    fx = sweep_force(
      self.env, self.body_id, self.dofadr, step, self.timestep, self.decimation
    )
    fz = PRESS_FORCE_Z if step > 200 else 0.0
    self.env.unwrapped.sim.data.xfrc_applied[0, self.body_id, 0:3] = torch.tensor(
      [fx, 0.0, fz], device=self.device, dtype=torch.float32
    )
    self.step_count += 1
    if step % 100 == 0:
      probe = self.env.unwrapped.sim.data.xpos[0, self.body_id].cpu().numpy()
      print(f"step {step:5d} | probe x={probe[0]:+.2f} z={probe[2]:.2f}")
    return torch.zeros(self.env.unwrapped.action_space.shape, device=self.device)


@dataclasses.dataclass(frozen=True)
class Config:
  """Tactile sensor demo configuration."""

  viewer: Literal["auto", "native", "viser"] = "auto"
  """Viewer to use. "auto" picks native on a display, viser otherwise."""
  timestep: float = 0.002
  """MuJoCo physics timestep in seconds."""
  decimation: int = 1
  """Physics substeps per environment step. Environment step duration =
  timestep * decimation."""
  print_every: int = 20
  """Env steps between terminal printouts (and viz updates)."""
  no_sensor: bool = False
  """Build the scene without the tactile sensor (A/B benchmarking)."""
  taxel_size: float = 0.3
  """Debug sphere size as a fraction of the scene's meansize."""


def main(cfg: Config) -> None:
  configure_torch_backends()

  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  print("=" * 60)
  print("Tactile Sensor Demo")
  print(f"  Timestep   : {cfg.timestep}")
  print(
    f"  Decimation : {cfg.decimation} "
    f"(env step = {cfg.timestep * cfg.decimation:.4f} s)"
  )
  print("=" * 60)
  print()

  env_cfg = create_env_cfg(
    cfg.timestep,
    cfg.decimation,
    with_sensor=not cfg.no_sensor,
    debug_vis_scale=cfg.taxel_size,
  )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env)

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.viewer

  if cfg.no_sensor:
    policy = ZeroPolicy(env, device, cfg.timestep, cfg.decimation)
  else:
    policy = TactilePrintPolicy(
      env, device, cfg.timestep, cfg.decimation, cfg.print_every
    )

  if resolved_viewer == "native":
    print("Launching native viewer...")
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    print("Launching viser viewer...")
    ViserPlayViewer(env, policy).run()
  else:
    raise ValueError(f"Unknown viewer: {resolved_viewer}")

  env.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
