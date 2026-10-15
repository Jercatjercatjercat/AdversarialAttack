# Adversarial Attacks and Defences: A Survey

[Source](https://arxiv.org/abs/1810.00069)

This survey supplies the basic language for describing an attack. White-box access includes knowledge of a model and its gradients. A targeted attack aims at a specified result; an untargeted attack seeks an unwanted result without specifying a particular wrong output. Evasion changes inputs when a trained model is used, whereas poisoning changes its training data.

The survey covers multiple studies, mainly classification, so it has no single weather input type or deterministic/probabilistic weather setup. Our fixed-CNN experiments are white-box evasion attacks on processed weather inputs. Numerical forecast error takes the place of a wrong image label.

## Questions to develop while reading

- Which assumptions about attacker access are realistic for our saved CNN and for Otter?
- Which classification measures need replacing for numerical weather forecasts?
- Which defenses have been evaluated against an attacker that knows the defense?
