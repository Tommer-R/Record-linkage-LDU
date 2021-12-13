from main_ldu_hw import groups as groups1, hw, ldu, groups_to_df, to_presentation
from main_ldu_hw_2 import groups as groups2

exclusive_1 = []
exclusive_2 = []
groups_1_lengths = []
groups_2_lengths = []

for g in groups1:
    groups_1_lengths.append(len(g[0]) + len(g[1]))
    if g in groups2:
        pass
    else:
        exclusive_1.append(g)

for g in groups2:
    groups_2_lengths.append(len(g[0]) + len(g[1]))
    if g in groups1:
        pass
    else:
        exclusive_2.append(g)


exclusive_1_df = groups_to_df(hw, ldu, exclusive_1)
exclusive_2_df = groups_to_df(hw, ldu, exclusive_2)
groups_1_df = groups_to_df(hw, ldu, groups1)
groups_2_df = groups_to_df(hw, ldu, groups2)

exclusive_1_df = to_presentation(exclusive_1_df)
exclusive_2_df = to_presentation(exclusive_2_df)
groups_1_df = to_presentation(groups_1_df)
groups_2_df = to_presentation(groups_2_df)

