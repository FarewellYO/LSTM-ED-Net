
"""
@author: Jingwen Xue
@date: 2026-09-27
@description:
Generate elastic and viscoelastic synthetic seismic records for salt rock
and brine replacement models, calculate interface reflection coefficients
and two-way travel times, and compare seismic responses under different
physical modeling conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Path and Basic Parameter Configuration
if "__file__" in globals():
    output_dir = Path(__file__).resolve().parent
else:
    output_dir = Path.cwd()

dt = 0.001
f_m = 30.0
f_ref = 1.0
nfft = 2048

print(f"Sampling interval: {dt * 1000:.1f} ms")
print(f"Ricker dominant frequency: {f_m:.1f} Hz")
print(f"Reference frequency: {f_ref:.1f} Hz")
print(f"FFT points: {nfft}")


# Load Geological Models
salt_path = output_dir / "salt_model.npz"
brine_path = output_dir / "brine_model.npz"

if not salt_path.is_file():
    raise FileNotFoundError(f"Salt model not found: {salt_path}")

if not brine_path.is_file():
    raise FileNotFoundError(f"Brine model not found: {brine_path}")

salt_model = np.load(salt_path)
brine_model = np.load(brine_path)

print("\nGeological models loaded successfully.")


# Calculate Interface Depths and Two-Way Travel Times
def calculate_interfaces(model):
    interface_depth = []
    twt = []
    upper_lithology = []
    lower_lithology = []
    travel_time = 0.0

    for i in range(len(model["lithology"]) - 1):
        travel_time += model["thickness"][i] / model["vp"][i]

        interface_depth.append(model["bottom_depth"][i])
        twt.append(2.0 * travel_time)
        upper_lithology.append(model["lithology"][i])
        lower_lithology.append(model["lithology"][i + 1])

    return (
        np.array(interface_depth),
        np.array(twt),
        np.array(upper_lithology),
        np.array(lower_lithology))


salt_depth, salt_twt, salt_upper, salt_lower = calculate_interfaces(salt_model)
brine_depth, brine_twt, brine_upper, brine_lower = calculate_interfaces(brine_model)


# Calculate Elastic Reflection Coefficients
def calculate_elastic_reflection(model):
    r0 = []

    for i in range(len(model["lithology"]) - 1):
        z1 = model["rho"][i] * model["vp"][i]
        z2 = model["rho"][i + 1] * model["vp"][i + 1]

        r0.append((z2 - z1) / (z2 + z1))

    return np.array(r0)


salt_r0 = calculate_elastic_reflection(salt_model)
brine_r0 = calculate_elastic_reflection(brine_model)


# Calculate Frequency-Dependent Complex Reflection Coefficients
def calculate_viscoelastic_reflection(freq, model, interface_index):
    rho1 = model["rho"][interface_index]
    rho2 = model["rho"][interface_index + 1]
    vp1 = model["vp"][interface_index]
    vp2 = model["vp"][interface_index + 1]
    q1 = model["q"][interface_index]
    q2 = model["q"][interface_index + 1]

    r0 = (rho2 * vp2 - rho1 * vp1) / (rho2 * vp2 + rho1 * vp1)
    dq = 1.0 / q2 - 1.0 / q1

    R = np.full(freq.shape, r0, dtype=complex)
    mask = freq > 0

    R[mask] = (
        r0
        + (1.0 / (2.0 * np.pi))
        * dq
        * np.log(np.abs(freq[mask] / f_ref))
        + 1j * 0.25 * np.abs(dq))

    return R


# Generate Ricker Wavelet and Its Spectrum
time_wavelet = (np.arange(nfft) - nfft // 2) * dt

ricker = (
    (1.0 - 2.0 * (np.pi * f_m * time_wavelet) ** 2)
    * np.exp(-(np.pi * f_m * time_wavelet) ** 2))

ricker_shifted = np.fft.ifftshift(ricker)
W = np.fft.rfft(ricker_shifted)
freq = np.fft.rfftfreq(nfft, d=dt)

print(f"\nFrequency resolution: {freq[1] - freq[0]:.4f} Hz")
print(f"Nyquist frequency: {freq[-1]:.1f} Hz")


# Generate Elastic Synthetic Seismic Records
def generate_elastic_record(model, twt):
    S = np.zeros_like(W, dtype=complex)
    r0 = calculate_elastic_reflection(model)

    for i in range(len(twt)):
        phase = np.exp(-1j * 2.0 * np.pi * freq * twt[i])
        S += W * r0[i] * phase

    S[0] = 0.0

    if nfft % 2 == 0:
        S[-1] = S[-1].real + 0j

    trace = np.fft.irfft(S, n=nfft)

    return trace


salt_elastic = generate_elastic_record(salt_model, salt_twt)
brine_elastic = generate_elastic_record(brine_model, brine_twt)


# Generate Synthetic Seismic Records Considering Dispersion and Attenuation
def generate_viscoelastic_record(model, twt):
    S = np.zeros_like(W, dtype=complex)

    for i in range(len(twt)):
        R = calculate_viscoelastic_reflection(freq, model, i)
        phase = np.exp(-1j * 2.0 * np.pi * freq * twt[i])
        S += W * R * phase

    S[0] = 0.0

    if nfft % 2 == 0:
        S[-1] = S[-1].real + 0j

    trace = np.fft.irfft(S, n=nfft)

    return trace


salt_viscoelastic = generate_viscoelastic_record(salt_model, salt_twt)
brine_viscoelastic = generate_viscoelastic_record(brine_model, brine_twt)

time_s = np.arange(nfft) * dt
time_ms = time_s * 1000.0


# Save Synthetic Seismic Records
def save_record(filename, trace, interface_depth, interface_twt, r0):
    np.savez(
        output_dir / filename,
        time_s=time_s,
        time_ms=time_ms,
        trace=trace,
        dt=dt,
        f_m=f_m,
        f_ref=f_ref,
        nfft=nfft,
        interface_depth=interface_depth,
        interface_twt_s=interface_twt,
        interface_twt_ms=interface_twt * 1000.0,
        r0=r0)


save_record("salt_elastic.npz", salt_elastic, salt_depth, salt_twt, salt_r0)
save_record("salt_viscoelastic.npz", salt_viscoelastic, salt_depth, salt_twt, salt_r0)
save_record("brine_elastic.npz", brine_elastic, brine_depth, brine_twt, brine_r0)
save_record("brine_viscoelastic.npz", brine_viscoelastic, brine_depth, brine_twt, brine_r0)

print("\nSynthetic records saved successfully:")
print(output_dir / "salt_elastic.npz")
print(output_dir / "salt_viscoelastic.npz")
print(output_dir / "brine_elastic.npz")
print(output_dir / "brine_viscoelastic.npz")


# Check Interface Parameters
def print_interfaces(title, depth, twt, upper, lower, r0):
    print(f"\n===== {title} =====")

    print(
        f"{'No.':<6}"
        f"{'Depth/m':<12}"
        f"{'TWT/ms':<12}"
        f"{'Upper':<24}"
        f"{'Lower':<24}"
        f"{'R0':<12}")

    for i in range(len(depth)):
        print(
            f"{i + 1:<6}"
            f"{depth[i]:<12.1f}"
            f"{twt[i] * 1000:<12.3f}"
            f"{upper[i]:<24}"
            f"{lower[i]:<24}"
            f"{r0[i]:<12.6f}")


print_interfaces(
    "Salt model interfaces",
    salt_depth, salt_twt, salt_upper, salt_lower, salt_r0)

print_interfaces(
    "Brine model interfaces",
    brine_depth, brine_twt, brine_upper, brine_lower, brine_r0)


# Quick Inspection of Synthetic Seismic Records
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

plot_mask = (time_ms >= 40) & (time_ms <= 260)

fig, axes = plt.subplots(1, 2, figsize=(7, 7), sharey=True)

axes[0].plot(
    salt_elastic[plot_mask],
    time_ms[plot_mask],
    linewidth=1.0,
    label="Elastic")

axes[0].plot(
    salt_viscoelastic[plot_mask],
    time_ms[plot_mask],
    linewidth=1.0,
    label="Viscoelastic")

axes[0].set_title("(a) Salt rock model")
axes[0].set_xlabel("Amplitude")
axes[0].set_ylabel("Time / ms")
axes[0].legend(frameon=False)

axes[1].plot(
    brine_elastic[plot_mask],
    time_ms[plot_mask],
    linewidth=1.0,
    label="Elastic")

axes[1].plot(
    brine_viscoelastic[plot_mask],
    time_ms[plot_mask],
    linewidth=1.0,
    label="Viscoelastic")

axes[1].set_title("(b) Brine model")
axes[1].set_xlabel("Amplitude")
axes[1].legend(frameon=False)

axes[0].set_ylim(260, 40)

plt.tight_layout()
plt.show()
