


fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize=(8,6))
ax1, ax2, ax3, ax4 = ax.flatten()


ax1.plot(grammages, long_profile_1, color = "darkblue", label = "S1", alpha=0.4)
ax1.plot(grammages, long_profile_2, color = "red", label = "S2", alpha = 0.4)
ax1.plot(grammages, long_profile_2 + long_profile_1, color = "green", label = "DB", lw = 1.8)
ax1.legend()
ax1.set_xlabel(r"$X \ [g/cm^2]$")
ax1.set_ylabel(r"$N(x)$")

ax2.plot(time_traces_per_antenna, geomagnetic_antenna_ef_1, color = "darkblue", label = f"S1", alpha = 0.5)
ax2.plot(time_traces_per_antenna, geomagnetic_antenna_ef_2, color = "red", label = f"S2", alpha = 0.5)
ax2.plot(time_traces_per_antenna, geomagnetic_antenna_ef_2 + geomagnetic_antenna_ef_1, color = "green", label = "DB", lw = 1.8)
ax2.set_xlim(-180, -140)
ax2.set_xlabel("t")
ax2.set_ylabel(r"$E_{geo}$")
ax2.set_title(f"antenna position: {antenna_position}, version: JAX")
ax2.legend()
