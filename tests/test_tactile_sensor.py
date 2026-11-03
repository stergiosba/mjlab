"""Tests for tactile_sensor.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor.builtin_sensor import BuiltinSensor, ObjRef
from mjlab.sensor.tactile_sensor import TactileSensor, TactileSensorCfg
from mjlab.sim.sim import Simulation, SimulationCfg


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def tactile_scene_xml():
  """XML for a tactile sensor geom pressing into a box (mirrors mujoco_warp's
  own tactile test setup)."""
  return """
    <mujoco>
      <option>
        <flag multiccd="enable"/>
      </option>
      <asset>
        <mesh name="sensor_mesh" builtin="sphere" params="0"/>
      </asset>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>
        <body name="sensor_body" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="sensor_geom" type="mesh" mesh="sensor_mesh"/>
        </body>
        <body name="box_body" pos="0 0 -0.3">
          <geom name="box_geom" type="box" size="0.7 0.7 0.3"/>
        </body>
      </worldbody>
    </mujoco>
  """


def test_tactile_sensor(tactile_scene_xml, device):
  """Verify tactile sensor outputs per-taxel data and reads pressure on
  contact."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(tactile_scene_xml))

  tactile_sensor_cfg = TactileSensorCfg(
    name="tactile",
    obj=ObjRef(type="mesh", name="sensor_mesh", entity="robot"),
    ref=ObjRef(type="geom", name="sensor_geom", entity="robot"),
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=3.0,
    entities={"robot": entity_cfg},
    sensors=(tactile_sensor_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=40)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["robot/tactile"]
  assert isinstance(sensor, TactileSensor)
  assert isinstance(sensor, BuiltinSensor)
  mj_sensor = sim.mj_model.sensor("robot/tactile")
  assert mj_sensor.type == mujoco.mjtSensor.mjSENS_TACTILE
  # One taxel per mesh vertex, with depth and two slip channels.
  mesh_id = mj_sensor.objid[0]
  num_taxels = sim.mj_model.mesh_vertnum[mesh_id]
  channels = 3
  assert mj_sensor.dim[0] == channels * num_taxels

  # Step until the sphere falls and presses into the box, generating contact.
  for _ in range(300):
    sim.step()

  data = sensor.data
  assert isinstance(data, torch.Tensor)
  assert data.shape == (2, channels * num_taxels)
  assert torch.any(data > 0), "Taxel readings should be nonzero on contact"


def test_tactile_sensor_channel_major_layout(tactile_scene_xml, device):
  """Tactile output is channel-major: all taxel depths first, then all slip
  values. Verify active (depth > 0) taxels are the bottom vertices of the
  pressing sphere, not arbitrary mesh indices."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(tactile_scene_xml))
  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=3.0,
    entities={"robot": entity_cfg},
    sensors=(
      TactileSensorCfg(
        name="tactile",
        obj=ObjRef(type="mesh", name="sensor_mesh", entity="robot"),
        ref=ObjRef(type="geom", name="sensor_geom", entity="robot"),
        debug_vis=True,
      ),
    ),
  )
  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(njmax=40), model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  sensor = scene["robot/tactile"]

  for _ in range(300):
    sim.step()

  data = sensor.data[0].cpu().numpy()
  num_taxels = int(
    sim.mj_model.mesh_vertnum[
      mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_MESH, "sensor_mesh")
    ]
  )
  channels = data.shape[0] // num_taxels
  assert channels == 3

  depths = data.reshape(channels, num_taxels)[0]
  active = np.flatnonzero(depths > 1e-6)
  assert active.size > 0

  # Bottom vertices of the sphere are the deepest ones: depth must decrease
  # with the taxel's local z (sphere mesh is centered at the origin).
  z = sim.mj_model.mesh_vert[:num_taxels, 2]
  corr = np.corrcoef(z, depths)[0, 1]
  assert corr < -0.5, f"depths should be largest for lowest vertices (corr={corr:.3f})"

  assert isinstance(sensor, TactileSensor)
  visualizer = MagicMock()
  visualizer.get_env_indices.return_value = [0]
  visualizer.meansize = 1.0
  sensor.debug_vis(visualizer)
  assert visualizer.add_sphere.call_count == active.size
  sensor._debug_vis_enabled = False
  visualizer.add_sphere.reset_mock()
  sensor.debug_vis(visualizer)
  visualizer.add_sphere.assert_not_called()

  # The interleaved (taxel-major) interpretation would mix depth and slip
  # values, so it must not reproduce the depth signal either.
  bad = data.reshape(num_taxels, channels)[:, 0]
  bad_active = np.flatnonzero(bad > 1e-6)
  if bad_active.size:
    assert bad_active.tolist() != active.tolist(), (
      "taxel-major reshape should not identify the same active taxels"
    )


def test_tactile_sensor_requires_mesh_and_geom(tactile_scene_xml):
  """Verify tactile sensor validation requires obj=mesh and ref=geom."""
  with pytest.raises(ValueError, match="requires obj with type='mesh'"):
    TactileSensorCfg(
      name="tactile",
      obj=ObjRef(type="geom", name="sensor_geom"),
      ref=ObjRef(type="geom", name="sensor_geom"),
    )

  with pytest.raises(ValueError, match="requires ref with type='geom'"):
    TactileSensorCfg(
      name="tactile",
      obj=ObjRef(type="mesh", name="sensor_mesh"),
      ref=ObjRef(type="site", name="sensor_site"),
    )
