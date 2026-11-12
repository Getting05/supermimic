# Astro P2 training asset

`mjcf/astro_p2_supermimic.xml` and `meshes/` were copied from docker150:
`/root/code/SuperMimic-Reproduced/src/supermimic/assets/docker150/astro_p2/`.
The XML and referenced meshes are preserved; the task factory adds the named
IMU sensors required by mjlab. No server-specific asset path is needed at runtime.

`p2_deploy_contract.json` comes from Getting05/SuperMimic-Reproduced main
(`4bfc8e25d10fb4df6490eb7567af5e5e97dd45ab`). Its joint names define the 29
controlled joints (excluding the final head joint); the task's PD gains and action
scales follow that repository's `envs/p2.py` formulas, not the JSON gain table.

P2 configuration and SGS/retarget algorithms were ported from that repository.
