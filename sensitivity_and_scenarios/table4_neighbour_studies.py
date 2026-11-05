"""Table 4, last two columns: for each selected candidate, the number of training observations within a
scaled distance of 1.0 in the standardised eight-dimensional support space, the number of distinct source
studies contributing them, and the number of distinct studies among the five nearest neighbours used in
the local-consistency check (Equation 9). Output: results/sensitivity/table4_neighbour_studies.csv"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from common import OUT, TABLE4, eng

ctx = eng.load_context(False)
study = eng.load_study_map(ctx)["study"].to_numpy()
train = ctx.train_ad_scaled
rows = []
for t, fa, ba, wb in TABLE4:
    ba = max(ba, float(ctx.df["B/A"].min()))  # 0.116 in Table 4 is the data minimum, rounded
    d = cdist(ctx.ad_scaler.transform(eng.ad_input_matrix(ctx, fa, ba, wb)), train)[0]
    k5 = np.argsort(d)[:eng.LOCAL_K]
    near = d <= 1.0
    rows.append(dict(target_MPa=t, nn_distance=round(float(d.min()), 3),
                     observations_within_1=int(near.sum()), studies_within_1=len(set(study[near])),
                     studies_among_5_nn=len(set(study[k5]))))
out = pd.DataFrame(rows)
out.to_csv(OUT / "table4_neighbour_studies.csv", index=False)
print(out.to_string(index=False))
