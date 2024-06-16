import numpy as np
from scipy import stats

# Let's assume these are your data
age_smokers = np.array([...])  # ages of mothers who smoke
age_non_smokers = np.array([...])  # ages of mothers who do not smoke

# Compute the observed test statistic
observed_tt = stats.ttest_ind(age_smokers, age_non_smokers).statistic

# Combine the data
all_ages = np.concatenate([age_smokers, age_non_smokers])

# Perform the permutation test
n_permutations = 10000
permuted_tt = np.zeros(n_permutations)
for i in range(n_permutations):
    permuted_ages = np.random.permutation(all_ages)
    permuted_smokers = permuted_ages[:len(age_smokers)]
    permuted_non_smokers = permuted_ages[len(age_smokers):]
    permuted_tt[i] = stats.ttest_ind(permuted_smokers, permuted_non_smokers).statistic

# Compute the p-value
p_value = np.mean(np.abs(permuted_tt) >= np.abs(observed_tt))

print(f'p-value = {p_value}')
