**Table 1.** Within-clade change per +10 °C of optimal growth temperature in the catalytic neighbourhood (10 Å) and in a size- and burial-matched control patch on the same protein, for the descriptors surviving Benjamini-Hochberg correction. Slopes are in each descriptor's own units; the SD column restates the difference in units of that descriptor's within-clade standard deviation. n = 10,674 proteins in 1,485 clades, AlphaFold models.

| Descriptor | Pocket | Control | Difference [95% CI] | Difference (SD) | q |
| --- | --- | --- | --- | --- | --- |
| Polar fraction (×10⁻³) | −0.200 | −4.099 | +3.899 [+1.924, +5.628] | +0.127 | < 0.001 |
| Proline fraction (×10⁻³) | +0.006 | +1.086 | −1.080 [−1.818, −0.347] | −0.092 | < 0.001 |
| IVYWREL fraction (×10⁻³) | +0.871 | +5.731 | −4.860 [−6.439, −3.299] | −0.149 | < 0.001 |
| Mean residue volume (Å³) | +0.012 | +0.243 | −0.231 [−0.349, −0.115] | −0.093 | 0.002 |
| Acidic fraction (×10⁻³) | +0.054 | +1.447 | −1.393 [−2.404, −0.420] | −0.074 | 0.011 |
| Small-residue fraction (×10⁻³) | −0.467 | −2.394 | +1.926 [+0.482, +3.447] | +0.068 | 0.016 |
| Aromatic fraction (×10⁻³) | −0.533 | +0.455 | −0.988 [−1.896, −0.062] | −0.066 | 0.024 |

**Table 2.** Proteins entering the analysis, by family and by domain. Counts are after structure, confidence and clade filters, so they are smaller than the number of sequences retrieved. A clade is an MMseqs2 cluster at 50% identity; thermophile means an optimal growth temperature of at least 50 °C.

| Structures | Group | Pfam | Proteins | Clades | Clades with a thermophile | Median OGT (°C) | OGT range (°C) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AlphaFold | S8 | PF00082 | 6,216 | 960 | 187 | 30 | 5–106 |
| AlphaFold | S9 | PF00326 | 3,332 | 374 | 67 | 30 | 5–105 |
| AlphaFold | S1 | PF00089 | 1,126 | 153 | 11 | 28 | 4–65 |
| AlphaFold | Bacteria | — | 10,046 | 1,404 | 211 | 30 | 4–80 |
| AlphaFold | Archaea | — | 628 | 88 | 54 | 40 | 22–106 |
| ESMFold | S8 | PF00082 | 3,147 | 256 | 90 | 30 | 7–106 |
| ESMFold | S9 | PF00326 | 955 | 61 | 21 | 30 | 5–80 |
| ESMFold | S1 | PF00089 | 1,343 | 74 | 10 | 28 | 4–58 |
| ESMFold | Bacteria | — | 5,291 | 370 | 105 | 30 | 4–80 |
| ESMFold | Archaea | — | 154 | 24 | 16 | 40 | 30–106 |
