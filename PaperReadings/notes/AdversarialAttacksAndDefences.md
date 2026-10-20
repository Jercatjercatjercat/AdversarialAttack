# Adversarial Attacks and Defences: A Survey

Read and notes: 19 October 2026.

- Mostly classification-specific methods; limited additional material for our weather project beyond the newer survey.
- **Black-box surrogate attacks:** train an accessible local model, craft attacks on it and test transfer to the target.
- **Non-adaptive:** train the local model using data from the assumed distribution μ, without adapting queries to the target.
- **Adaptive:** query the target to label carefully selected inputs and train the local model. This is the survey's terminology, not a requirement for all black-box attacks; some search directly through queries without a surrogate.
- Here, “adaptive” concerns target querying, distinct from adapting an attack to a known defense.

## Bookmark

- Black-box access assumptions and surrogate-model transfer, if we extend beyond white-box CNN attacks.
