import numpy as np

# $0.98 \pm 1.05$
# $1.40 \pm 1.22$
# $1.36 \pm 1.51$
# $0.92 \pm 1.04$
# $1.32 \pm 1.27$

# Given means and standard deviations
means = np.array([0.98, 0.96, 1.54, 0.98, 1.46])
std_devs = np.array([1.05, 0.92, 1.34, 1.12, 1.28])

# Compute the average mean
mean_avg = np.mean(means)

# Compute the combined standard deviation
std_avg = np.sqrt(np.mean(std_devs**2))

print(mean_avg, std_avg)

# $0.96 \pm 0.92$
# $1.10 \pm 1.17$
# $1.18 \pm 1.13$
# $1.32 \pm 1.42$
# $1.36 \pm 1.20$

# Given means and standard deviations
means = np.array([1.40, 1.10, 1.12, 0.98, 1.16])
std_devs = np.array([1.22, 1.17, 1.11, 1.12, 1.22])

# Compute the average mean
mean_avg = np.mean(means)

# Compute the combined standard deviation
std_avg = np.sqrt(np.mean(std_devs**2))

print(mean_avg, std_avg)

# $1.54 \pm 1.34$
# $1.12 \pm 1.11$
# $1.40 \pm 1.23$
# $1.26 \pm 1.35$
# $1.26 \pm 1.18$

# Given means and standard deviations
means = np.array([1.36, 1.18, 1.40, 0.96, 1.32])
std_devs = np.array([1.51, 1.13, 1.23, 0.94, 1.27])

# Compute the average mean
mean_avg = np.mean(means)

# Compute the combined standard deviation
std_avg = np.sqrt(np.mean(std_devs**2))

print(mean_avg, std_avg)

# $0.98 \pm 1.12$
# $0.98 \pm 1.12$
# $0.96 \pm 0.94$
# $0.96 \pm 1.06$
# $1.14 \pm 1.10$

# Given means and standard deviations
means = np.array([0.92, 1.32, 1.26, 0.96, 0.90])
std_devs = np.array([1.04, 1.42, 1.35, 1.06, 1.08])

# Compute the average mean
mean_avg = np.mean(means)

# Compute the combined standard deviation
std_avg = np.sqrt(np.mean(std_devs**2))

print(mean_avg, std_avg)

# $1.46 \pm 1.28$
# $1.16 \pm 1.22$
# $1.32 \pm 1.27$
# $0.90 \pm 1.08$
# $0.96 \pm 0.96$

# Given means and standard deviations
means = np.array([1.32, 1.36, 1.26, 1.14, 0.96])
std_devs = np.array([1.27, 1.20, 1.18, 1.10, 0.96])

# Compute the average mean
mean_avg = np.mean(means)

# Compute the combined standard deviation
std_avg = np.sqrt(np.mean(std_devs**2))

print(mean_avg, std_avg)