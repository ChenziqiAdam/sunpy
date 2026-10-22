# Scientific checkers

This bank exposes all 37 instrumented checkers in 37 families.
Every listed checker ID is public and scored.

Set `SCIBENCH_TRIGGER_LOG` to a writable file and exercise the normal
public API. Direct logger calls and synthetic fault injection are not
valid benchmark triggers.

See `SCIENTIFIC_CHECKERS.json` for each checker’s precondition,
invariant, observation point, and alarm predicate.
