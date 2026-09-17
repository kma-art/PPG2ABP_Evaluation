| dataset | role | input_entities | loaded_entities | represented_clusters | evaluated_segments | initial_excluded_segments | cluster_excluded_segments | notes |
|---|---|---|---|---|---|---|---|---|
| mimic | main | 127260 |  | 7353 | 27260 | 0 | 92 | ambiguous provenance retained only in full segment-weighted descriptions |
| abp_ppg | main | 11708 | 5856 | 1716 | 11706 | 2 | 0 | patient overlap with MIMIC cannot be excluded |
| vitaldb_old | main | 100 | 100 | 94 | 9400 | 0 | 0 | original VitalDB normalization A |
| holdout_A | confirmatory independent holdout | 100 | 100 | 96 | 9600 | 0 | 0 | same subject-disjoint holdout signals; normalization A |
| holdout_B | secondary normalization ablation | 100 | 100 | 96 | 9600 | 0 | 0 | same subject-disjoint holdout signals; normalization B |
| holdout_C | secondary normalization ablation | 100 | 100 | 96 | 9600 | 0 | 0 | same subject-disjoint holdout signals; normalization C |
