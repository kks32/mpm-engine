"""Function-encoder TRAINING (Mode F, Phase 3).

This is the ONLY place under ident/ where torch may be imported (project
invariant 2). Nothing in the Phase 1/2 identification path imports this
package; the deployed artifact is a frozen tabulation wrapped by
ident.features.function_encoder.FunctionEncoderDict, which is pure numpy.

The training learns a BASIS that spans a corpus of mu(I) laws by closed-form
projection (docs/FUNCTION_ENCODER.md Section 2). It does NOT differentiate the
simulator and it does NOT do rollout matching (invariant 1): the basis is
trained offline on mu(I) functions, then used in the SAME linear weak-form
solve as Modes C and P.
"""
