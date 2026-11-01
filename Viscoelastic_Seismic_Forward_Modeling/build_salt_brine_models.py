'''
@author: Jingwen Xue
@date: 2026-09-27
@description: 
This script constructs and visualizes geological models for rock salt and brine replacement scenarios, assigns lithological properties, and exports the numerical model data for subsequent viscoelastic seismic forward modeling.
'''

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

#%% Path and Parameter Configuration
base_dir = r"The_path_to_working_directory"
output_dir = os.path.join(base_dir, "model_data")
os.makedirs(output_dir, exist_ok=True)

properties = {"Gypsiferous limestone": [4400.0, 3100.0, 85.0],
    "Gypsum": [5390.0, 2780.0, 120.0],
    "Salt rock": [4300.0, 2300.0, 100.0],
    "Brine": [1553.0, 1170.0, 1000.0]}

#%% Build Stratigraphic Model
def build_model(target_lithology):
    layers = []
    layers.append(("Gypsiferous limestone", 150.0))
    layers.append(("Gypsum", 70.0))

    for i in range(8):
        layers.append((target_lithology, 10.0))
        if i < 7:
            layers.append(("Gypsum", 10.0))

    layers.append(("Gypsiferous limestone", 150.0))
    return layers

salt_layers = build_model("Salt rock")
brine_layers = build_model("Brine")

#%% Convert to Numerical Model
def convert_model(layers):
    lithology, top_depth, bottom_depth, thickness, vp, rho, q = [], [], [], [], [], [], []
    depth = 0.0

    for name, h in layers:
        top = depth
        bottom = depth + h
        lithology.append(name)
        top_depth.append(top)
        bottom_depth.append(bottom)
        thickness.append(h)
        vp.append(properties[name][0])
        rho.append(properties[name][1])
        q.append(properties[name][2])
        depth = bottom

    return {
        "lithology": np.array(lithology),
        "top_depth": np.array(top_depth, dtype=float),
        "bottom_depth": np.array(bottom_depth, dtype=float),
        "thickness": np.array(thickness, dtype=float),
        "vp": np.array(vp, dtype=float),
        "rho": np.array(rho, dtype=float),
        "q": np.array(q, dtype=float)
    }

salt_model = convert_model(salt_layers)
brine_model = convert_model(brine_layers)

#%% Save Numerical Models
def save_model(model, filename):
    np.savez(os.path.join(output_dir, filename), **model)


save_model(salt_model, "salt_model.npz")
save_model(brine_model, "brine_model.npz")

#%% Inspect Numerical Models
def print_model(model, title):
    print(f"\n===== {title} =====")
    print(f"{'No.':<6}{'Top Depth/m':<12}{'Bottom Depth/m':<15}{'Thickness/m':<12}{'Lithology':<24}{'Vp/(m/s)':<12}{'Density/(kg/m3)':<18}{'Q':<8}")

    for i in range(len(model["lithology"])):
        print(f"{i+1:<6}{model['top_depth'][i]:<10.1f}{model['bottom_depth'][i]:<10.1f}{model['thickness'][i]:<10.1f}{model['lithology'][i]:<10}{model['vp'][i]:<12.1f}{model['rho'][i]:<15.1f}{model['q'][i]:<8.1f}")

    print(f"Total model thickness: {model['bottom_depth'][-1]:.1f} m")


print_model(salt_model, "Salt Rock Model")
print_model(brine_model, "Brine Replacement Model")

print("\nModels saved successfully:")
print(os.path.join(output_dir, "salt_model.npz"))
print(os.path.join(output_dir, "brine_model.npz"))

#%% Visualize Geological Models
figure_dir = os.path.join(base_dir, "figures")
os.makedirs(figure_dir, exist_ok=True)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

x_length = 100.0
x_ticks = np.arange(0, 101, 20)
y_ticks = np.arange(0, 501, 50)

lithology_en = {"Gypsiferous limestone": "Gypsiferous limestone",
    "Gypsum": "Gypsum",
    "Salt rock": "Salt rock",
    "Brine": "Brine"}

colors = {"Gypsiferous limestone": "#6A9BE3",
    "Gypsum": "#8A5D8F",
    "Salt rock": "#F8EFD8",
    "Brine": "#71B7A8"}

def plot_geological_model(ax, model, title):
    for i in range(len(model["lithology"])):
        name = model["lithology"][i]
        top = model["top_depth"][i]
        thickness = model["thickness"][i]
        ax.barh(top, x_length, height=thickness, left=0, align="edge", color=colors[name], edgecolor="black", linewidth=0.6)

    ax.set_xlim(0, x_length)
    ax.set_ylim(520, 0)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xlabel("Horizontal distance / m", fontsize=10)
    ax.set_ylabel("Depth / m", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(False)

fig, axes = plt.subplots(1, 2, figsize=(8, 7), sharey=True)

plot_geological_model(axes[0], salt_model, "(a) Salt rock model")
plot_geological_model(axes[1], brine_model, "(b) Brine model")

axes[1].set_ylabel("")

legend_order = ["Gypsiferous limestone","Gypsum","Salt rock","Brine"]
legend_handles = [Patch(facecolor=colors[name], edgecolor="#555555", label=lithology_en[name]) for name in legend_order]
fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, fontsize=9)

plt.tight_layout(rect=[0, 0.06, 1, 1])

figure_path = os.path.join(figure_dir, "geological_model_v00.tif")
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nGeological model figure saved to:\n{figure_path}")