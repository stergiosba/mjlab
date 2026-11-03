"""MuJoCo tactile sensor and taxel visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.sensor.builtin_sensor import BuiltinSensor, ObjRef
from mjlab.sensor.sensor import SensorCfg

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass
class TactileSensorCfg(SensorCfg):
  obj: ObjRef
  """Taxel mesh reference."""
  ref: ObjRef
  """Reference to the geom carrying the taxel mesh."""
  cutoff: float = 0.0
  """When positive, limits the absolute value of the sensor output."""
  debug_vis: bool = False
  """Draw active taxels, colored by penetration depth."""
  debug_vis_scale: float = 0.05
  """Taxel sphere size as a fraction of the scene's ``meansize``."""

  @property
  def prefixed_name(self) -> str:
    if self.obj.entity is not None:
      return f"{self.obj.entity}/{self.name}"
    return self.name

  def __post_init__(self) -> None:
    if self.obj.type != "mesh":
      raise ValueError("Tactile sensor requires obj with type='mesh'")
    if self.ref.type != "geom":
      raise ValueError("Tactile sensor requires ref with type='geom'")

  def build(self) -> TactileSensor:
    return TactileSensor(self)


class TactileSensor(BuiltinSensor):
  """Builtin tactile sensor with per-taxel debug visualization."""

  def __init__(self, cfg: TactileSensorCfg) -> None:
    super().__init__(name=cfg.prefixed_name)
    self.tactile_cfg = cfg
    self._debug_vis_enabled = True
    self._debug_taxels: tuple[torch.Tensor, int, int, int] | None = None

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    del entities
    cfg = self.tactile_cfg
    if any(sensor.name == cfg.prefixed_name for sensor in scene_spec.sensors):
      raise ValueError(f"Sensor '{cfg.prefixed_name}' already exists in the scene")
    scene_spec.add_sensor(
      name=cfg.prefixed_name,
      type=mujoco.mjtSensor.mjSENS_TACTILE,
      objtype=mujoco.mjtObj.mjOBJ_MESH,
      objname=cfg.obj.prefixed_name(),
      reftype=mujoco.mjtObj.mjOBJ_GEOM,
      refname=cfg.ref.prefixed_name(),
      cutoff=cfg.cutoff if cfg.cutoff > 0 else None,
    )

  def initialize(
    self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str
  ) -> None:
    super().initialize(mj_model, model, data, device)
    sensor = mj_model.sensor(self._name)
    mesh_id = int(sensor.objid[0])
    geom_id = int(sensor.refid[0])
    vertadr = mj_model.mesh_vertadr[mesh_id]
    num_taxels = int(mj_model.mesh_vertnum[mesh_id])
    verts = torch.as_tensor(
      mj_model.mesh_vert[vertadr : vertadr + num_taxels],
      dtype=torch.float32,
      device=data.sensordata.device,
    )
    channels = int(sensor.dim[0]) // num_taxels if num_taxels > 0 else 0
    self._debug_taxels = (verts, num_taxels, channels, geom_id)

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if (
      not self.tactile_cfg.debug_vis
      or not self._debug_vis_enabled
      or self._debug_taxels is None
    ):
      return
    assert self._data is not None
    assert self._data_view is not None
    verts, num_taxels, channels, geom_id = self._debug_taxels
    if num_taxels == 0 or channels == 0:
      return

    env_indices = list(visualizer.get_env_indices(self._data_view.shape[0]))
    if not env_indices:
      return
    idx = torch.tensor(env_indices, device=self._data_view.device, dtype=torch.long)

    # Warp stores all taxel depths first, followed by the slip channels.
    readings = self._data_view[idx].reshape(-1, channels, num_taxels)
    depths = readings[:, 0, :]
    if channels == 1:
      depths = depths.abs()
    if not (depths > 1e-6).any():
      return

    geom_xpos = self._data.geom_xpos[idx, geom_id]
    geom_xmat = self._data.geom_xmat[idx, geom_id].view(-1, 3, 3)
    verts_w = torch.einsum("kij,nj->kni", geom_xmat, verts) + geom_xpos.unsqueeze(1)

    verts_w = verts_w.cpu().numpy()
    depths = depths.cpu().numpy()
    max_depth = depths.max()
    radius = self.tactile_cfg.debug_vis_scale * visualizer.meansize
    for k in range(len(env_indices)):
      for taxel in np.flatnonzero(depths[k] > 1e-6).tolist():
        scale = float(depths[k, taxel] / max_depth)
        visualizer.add_sphere(
          center=verts_w[k, taxel],
          radius=radius * (0.5 + scale),
          color=(1.0, 1.0 - scale, 1.0 - scale, 1.0),
        )
