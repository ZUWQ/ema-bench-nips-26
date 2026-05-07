# EMA-Bench: A Benchmark for Embodied Multi-Agent Decision-Making in Dynamic Environments

## Abstract

Embodied multi-agent systems are essential for high-risk disaster response, yet they remain limited in dynamic environments where hazards escalate rapidly and environmental states evolve in a path-dependent manner. In such scenarios, agents cannot rely solely on reactive execution. They must reason proactively about future environmental changes, prioritize time-critical objectives, and coordinate with other agents before the disaster outpaces the team's response capacity.

We introduce **EMA-Bench**, a high-fidelity simulation benchmark for evaluating embodied multi-agent coordination in self-progressing fire environments. EMA-Bench models disaster response as an interactive process in which agent actions directly influence environmental evolution under strict temporal urgency and partial observability. Agents must explore unknown regions, suppress spreading fires, reason about irreversible state transitions, and coordinate their actions across space and time.

To support systematic evaluation, EMA-Bench provides a structured framework covering foundational task execution, environmental exploration, and collaborative efficiency. Our empirical analysis of state-of-the-art multimodal foundation model-based agents reveals substantial weaknesses in handling time-sensitive trade-offs, irreversible dynamics, and multi-agent resource allocation. These results expose a critical gap in current embodied intelligence and establish EMA-Bench as a rigorous testbed for future research on resilient multi-agent decision-making.

## Background

Disaster response represents one of the most demanding application domains for embodied intelligence. In fire rescue, flood response, earthquake search-and-rescue, and other emergency scenarios, the environment changes continuously over time. Delayed or poorly coordinated actions can permanently block paths, destroy targets, or reduce the feasibility of subsequent rescue operations.

Compared with static navigation, object manipulation, or single-agent planning tasks, dynamic disaster environments introduce several distinctive challenges:

- **Rapid hazard escalation**: Fire, smoke, and structural damage can quickly alter traversable areas and task priorities.
- **Path-dependent dynamics**: Early decisions influence later environmental states, making local mistakes difficult or impossible to recover from.
- **Strict temporal constraints**: Agents must complete key interventions before the disaster reaches a critical threshold.
- **Partial observability**: Individual agents have limited views and must infer global conditions through exploration and communication.
- **Multi-agent coordination**: Complex disaster response tasks often exceed the capability of a single agent and require coordinated division of labor.

These properties make it insufficient to evaluate agents only by final task success. A meaningful benchmark must also measure whether agents can anticipate environmental evolution, intervene before irreversible failures occur, and coordinate effectively under time pressure.

## EMA-Bench Platform

EMA-Bench is designed to simulate dynamic fire response scenarios with self-progressing environmental dynamics. The benchmark evaluates how embodied agent teams perceive, reason, and act in environments where hazards continue to evolve regardless of agent readiness.

The platform is built around the following design principles:

- **Self-progressing fire dynamics**: Fires spread over time and can intensify if left unattended.
- **Action-environment coupling**: Agent actions such as movement, exploration, and fire suppression affect both immediate task progress and future environmental states.
- **Partial observability**: Agents operate with limited local observations and must build situational awareness through exploration and information sharing.
- **Time-sensitive objectives**: Task utility depends not only on whether an objective is completed, but also on when it is completed.
- **Multi-agent collaboration**: Multiple embodied agents act simultaneously, enabling evaluation of coordination, role assignment, and resource scheduling.

By combining these elements, EMA-Bench goes beyond testing whether agents can execute isolated skills. It evaluates whether they can understand dynamic risk, predict future constraints, and adapt team-level strategies as the environment changes.

## Evaluation Framework

EMA-Bench evaluates embodied multi-agent systems across three complementary dimensions.

### Foundational Task Execution

This dimension measures whether agents can perform basic embodied operations such as navigation, target localization, fire suppression, and rescue-related actions. It reflects the agents' ability to execute low-level tasks reliably within the simulation environment.

### Environmental Exploration and Understanding

This dimension evaluates whether agents can efficiently explore partially observed environments, identify fire sources, detect hazardous regions, locate critical paths, and reason about evolving scene states from incomplete information.

### Collaborative Efficiency

This dimension measures whether agent teams can divide responsibilities, avoid redundant actions, share useful information, and complete high-priority objectives before the fire spreads beyond control.

Together, these dimensions provide a process-oriented evaluation of decision quality, rather than relying only on aggregate success rates.

## Empirical Findings

We evaluate state-of-the-art multimodal foundation model-based agents on EMA-Bench and observe several recurring limitations:

- Agents often fail to identify high-priority risks in time when fire spreads rapidly.
- Models struggle to balance immediate task rewards against long-term environmental consequences.
- Agents show limited capability in planning around irreversible state transitions.
- Multi-agent teams frequently suffer from redundant exploration, delayed coordination, or conflicting task allocation.
- Current models can often perform short-horizon perception and action, but struggle to maintain coherent team-level strategies over time.

These findings suggest that current embodied agents remain far from reliable deployment in dynamic, high-risk disaster response settings.

## Contributions

This work makes the following contributions:

- We propose **EMA-Bench**, a high-fidelity benchmark for embodied multi-agent decision-making in dynamic disaster response environments.
- We introduce self-progressing fire dynamics that couple agent actions with environmental evolution.
- We design a structured evaluation framework covering foundational execution, environmental exploration, and collaborative efficiency.
- We provide an empirical analysis of multimodal foundation model-based agents under time-sensitive and irreversible dynamics.
- We establish a reproducible benchmark for future research on resilient embodied multi-agent coordination.

## Conclusion

EMA-Bench advances embodied multi-agent evaluation from static task completion toward dynamic, time-sensitive, and tightly coupled disaster response scenarios. It emphasizes proactive reasoning about environmental evolution, early intervention against irreversible risks, and effective team coordination under strict temporal constraints. By exposing the limitations of current foundation model-based agents, EMA-Bench provides a foundation for developing more reliable, anticipatory, and resilient embodied intelligence systems.
