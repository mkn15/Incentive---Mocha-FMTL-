Federated Learning facilitates collaborative model training while preserving data privacy, yet dynamic edge environments present formidable challenges from device heterogeneity, sporadic participation, and information staleness. Although
Stackelberg-based incentive mechanisms have shown promise in resource allocation, they predominantly target single-task settings with limited multi-task adaptation. This paper introduces a sophisticated Age of Information (AoI)-aware incentive framework
for Federated Multi-Task Learning (FMTL) that elevates data freshness to a pivotal strategic variable. The server dynamically calibrates incentives by considering both computational contributions and update timeliness within a Stackelberg game.

  Three principal contributions are presented. First, the Stackelberg equilibrium under AoI constraints is rigorously characterized, yielding closed-form strategies that optimize accuracy,
freshness, and budget. Second, convergence analysis proves that AoI-weighted aggregation bounds staleness propagation with formal guarantees. Third, an innovative payment mechanism
penalizes obsolete updates while rewarding timely participation. Extensive experiments on MNIST, FEMNIST, and HAR datasets validate the approach. AoI-MOCHA achieves 95.2%,
93.8%, and 92.8% accuracy, reduces average AoI to 2.8, 3.2, and 3.5 rounds, and converges in 25–32 rounds—40–44% faster than MOCHA and 70–75% faster than FedAvg. Incentive costs of $35–
$38 reflect a 53–54% reduction over MOCHA. All workers attain positive utility, and the Price of Anarchy reaches 1.35, indicating near-optimal social welfare. These findings demonstrate that
freshness-aware incentive design enhances convergence, accuracy, and participation while preserving privacy and economic feasibility under heterogeneous conditions.
