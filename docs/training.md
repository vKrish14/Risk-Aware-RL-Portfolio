# Training

Install requirements in the local virtual environment as shown in the README. Verify the downloaded datasets first:

```powershell
python -m src.data.pipeline
python -m src.training.train --timesteps 100000 --reward risk_aware
```

Train all reward variants under the same PPO budget:

```powershell
python -m src.training.train --reward all
```

`--timesteps` overrides the configured budget. SB3 collects complete rollouts (`n_steps=2048`), so actual timesteps can exceed a requested budget by less than one rollout; both counts are written to model metadata. PPO uses MlpPolicy, CPU, seed 42, learning rate 0.0003, gamma 0.99, GAE lambda 0.95, clip 0.2, batch 64, entropy coefficient 0.01, and value coefficient 0.5.

`EvalCallback` evaluates deterministically on validation observations and saves the highest validation-reward checkpoint under `models/checkpoints/<variant>/best/`. The final checkpoint is saved beside its metadata. Test prices are not used in checkpoint selection. Validation rewards are training diagnostics and are not portfolio performance claims.
