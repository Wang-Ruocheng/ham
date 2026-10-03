Generated using the built-in image_gen tool. Conceptual illustration, not measured data.
The schematic 16→8→4→2 chain is not the exact depth of the current 2D experiment.

Use case: scientific-educational.
Create a polished, publication-quality conceptual methods figure for a physics research paper about QCR, an ensemble-based recurrent renormalization-group neural network. White background, crisp flat scientific illustration, restrained blue/teal/orange palette, highly legible English typography, balanced generous whitespace, wide landscape aspect ratio. No glossy 3D, no decorative neural-network spaghetti, no watermark. The figure must be scientifically correct and easy to read.

Title: "QCR: Learning renormalization from ensembles".
Main panel occupying upper two thirds: a precise commuting ladder with FOUR aligned columns and TWO rows.
Top row headed "Configuration ensembles". At columns display small STACKS of three square black-and-white spin lattices representing D_0, D_1, D_2, D_3. Grid resolutions clearly shrink left to right (16x16, 8x8, 4x4, 2x2), but ensemble stacks remain stacks throughout. Three rightward teal arrows between stacks, each labeled "C". Caption over that row "Shared local coarse-graining". The binary grids are schematic, not experimental data.
Bottom row headed "Latent coupling coordinates". Four rounded blue nodes labeled g_0, g_1, g_2, g_3. Directly DOWNWARD blue arrows from each D_k stack to corresponding g_k node, each labeled "Q". Rightward orange arrows between successive g nodes, each labeled "R". This square-ladder expresses Q(C D) approximately equals R(Q(D)); do not draw Q acting on a single spin or reverse the arrows.
Below the bottom row draw ONE unobtrusive orange curved dashed bypass arrow from g_0 to g_3 labeled "R^3", clearly a multistep rollout, ending at g_3. Caption beside it "Match direct encoding and recurrent rollout". No double-headed arrows, no latent feedback into C.

Lower third: THREE small neatly aligned explanatory inset cards, each connected to the matching module by palette only, not cluttered arrows:
"C | Local coarse-grainer" with a 2x2 spin block mapping to one binary spin. Caption "Same kernel at every block and scale".
"Q | Ensemble encoder" with small patch feature icons merging into a single vector. Caption "Local features -> average over sites and samples -> g".
"R | Recurrent coupling map" with a simple two-node residual transformation graphic. Caption "g_{k+1} = g_k + F(g_k)" and "Same map at every scale".

At very bottom a narrow neutral footer:
"Training: subset consistency + data geometry + kernel anti-collapse + single-step and multi-step closure"
and a smaller line:
"Unlabeled ensembles in; learned coarse-graining and latent RG dynamics out."
This current formulation does NOT enforce unit covariance; do NOT mention isotropy, RSMI, Hamiltonian reconstruction, supervised coupling labels, universal critical exponent recovery, or exact physical RG.
Use small panel identifiers "a" for main ladder and "b" for the three module cards. Typography should look like a professional vector diagram for a journal, with correct g and D subscripts and consistent arrow semantics. Prioritize clarity over text density; faithfully render all math labels.
