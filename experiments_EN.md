# Experimenting with BioCortexAI

This document serves as a guide for experimenting with the BioCortexAI framework. The goal is to explore how different parameters and approaches affect the model's behavior and capabilities.

## 1. Tuning and Experimenting with PlantNet Behavior

Here, you become a biologist/psychologist studying the behavior of your "organism." Change the parameters and observe how its reactions evolve.

### Sensitivity and Reactivity
- **Global Mood Influence**: In `plant_net.py` (the `PlantCell.step` method), try increasing the `global_influence` variable. What happens when the overall "mood" has a greater impact than local interactions? Does the model become more stable or more "moody"?
- **Return-to-Baseline Speed**: In `config.py` (within `PERSONALITY_PROFILES`), adjust the `homeostasis_rate`. What happens if the model returns to its equilibrium very slowly? Will it remain "offended" or "excited" for longer? What if it returns very quickly?

### Memory and Personality
- **Memory Persistence**: In the `AssociativeMemory` class (`plant_net.py`), change the `decay_rate`. How does behavior change if the memory is nearly permanent (low `decay_rate`) versus very short-term (high `decay_rate`)?
- **Personality Drift Speed**: Accelerate the `_personality_drift` rate in `PlantCell`. You will see changes in the model's baseline behavior more quickly if you expose it to specific situations (e.g., only negative messages).

### Tissue Structure
- **Network Architecture**: In `config.py` (`PLANT_TISSUE_MAP`), design a completely different network structure. What happens if the network is composed mainly of `MemoryCell`s? Or if `SensorCell`s are only on the periphery?

> **Recommendation**: Create a copy of `config.py` for each experiment to easily compare the results.

## 2. Evaluation and Measurement

Try to objectively measure how the stateful model is better (or different) than the stateless one.

### Qualitative Evaluation (Human Feedback)
1.  Have several people chat with both models—the standard one (without PlantNet) and the stateful one.
2.  Don't tell them which is which (a blind test).
3.  Ask them which model seemed more "human-like," "consistent," "interesting," or "less repetitive." Record their observations.

### Quantitative Evaluation (Metrics)
- **Response Diversity**: Ask both models the same question 100 times (with `temperature > 0`). Measure how much the answers vary (e.g., using metrics like self-BLEU or by counting unique responses). The stateful model is expected to produce more diverse answers as its internal state changes.
- **Long-term Consistency**: Create a scenario where you refer back to something said much earlier in the conversation. Observe whether the stateful model (thanks to its memory and hormones) maintains context better.

## 3. Possible Extensions (Ideas for Future Work)

- **Hormonal Influence on Generation Parameters**: In addition to modulating internal computations, the global hormone state could directly influence generation parameters in `generate.py`.
  - **High Dopamine**: Automatically slightly increase `temperature`.
  - **High Cortisol**: Automatically decrease `temperature` and increase `repetition_penalty`.
- **More Complex Inputs for PlantNet**: Instead of averaging `hidden_states`, a small neural network (e.g., a simple MLP) could be used to extract a more relevant `context_vector` from the entire `hidden_state` tensor.
- **Visualization**: Create a script to graphically display the 2D cell grid, their hormonal levels, and the strength of their connections in real-time. This would be fascinating to watch during a conversation.