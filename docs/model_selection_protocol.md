# Model-Selection Protocol

Six predefined sparse candidates vary word or character representation and
regularization. For each seed, one selector maximizes exposed target
validation accuracy and a second selector maximizes ancestry-clean validation
accuracy. Ties follow the frozen candidate order.

The two selected candidates are then compared on the untouched clean test.
The observed clean-selected minus exposed-selected difference is +3.125
accuracy points. Because the three nominal sparse seeds produce identical
prediction vectors, the primary paired-context bootstrap interval is
[-0.98, 7.23] points and the exact McNemar two-sided p-value is 0.167.

This result demonstrates a selection disagreement and an observed downstream
difference, but the downstream advantage is not statistically resolved.
