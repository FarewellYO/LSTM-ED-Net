"""
@author: Jingwen Xue
@date: 2026-9-2
@description:
This script reconstructs a 3D salt cavern prediction volume, applies horizon-based spatial constraints, and visualizes the predicted cavern together with sonar measurements for validation.
"""

#%%
import os
import numpy as np
import pandas as pd
from skimage.measure import marching_cubes
import pyvista as pv

#%% Path Configuration
binary_dir = r"The_path_to_binary_prediction_data"
horizon_dir = r"The_path_to_horizon_data"
sonar_dir = r"The_path_to_sonar_data"

sonar_files = {
    "Sonar_1": "The_sonar_file_1.csv",
    "Sonar_2": "The_sonar_file_2.csv"}

#%% Study Area Parameters
inline_start = None  # Starting Inline number
inline_end = None    # Ending Inline number

cdp_start = None     # Starting Xline number
cdp_end = None       # Ending Xline number

time_start = None    # Starting time sample (ms)
time_end = None      # Ending time sample (ms)

dt = None            # Sampling interval (ms) 

if any(v is None for v in [
    inline_start, inline_end,
    cdp_start, cdp_end,
    time_start, time_end, dt]):
    raise ValueError("Please specify all study area parameters before running.")

if dt <= 0:
    raise ValueError("Sampling interval must be positive.")

n_inline = inline_end - inline_start + 1
n_cdp = cdp_end - cdp_start + 1
n_time = int(round((time_end - time_start) / dt)) + 1

#%% Load All Binary Predictions and Construct the Complete 3D Volume
binary_volume = np.zeros((n_inline, n_cdp, n_time), dtype=np.uint8)

for inline_idx in range(n_inline):

    file_name = f"inline_{inline_idx:03d}_binary.npy"
    file_path = os.path.join(binary_dir, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    section = np.load(file_path)

    if section.shape != (n_time, n_cdp):
        raise ValueError(f"Invalid shape for {file_name}: {section.shape}")

    binary_volume[inline_idx, :, :] = (section.T >= 0.5).astype(np.uint8)

print("Complete 3D volume shape:", binary_volume.shape)
print("Number of positive voxels in the complete 3D volume:", np.sum(binary_volume == 1))

#%% Load Upper and Lower Boundary Horizons
upper_time_map = np.load(os.path.join(horizon_dir, "The_upper_horizon_time_map.npy"))
lower_time_map = np.load(os.path.join(horizon_dir, "The_lower_horizon_time_map.npy"))

expected_shape = (n_inline, n_cdp)

if upper_time_map.shape != expected_shape:
    raise ValueError("Upper horizon time map has an invalid shape.")

if lower_time_map.shape != expected_shape:
    raise ValueError("Lower horizon time map has an invalid shape.")

#%% Validate Upper and Lower Horizon Boundaries
valid_horizon = (np.isfinite(upper_time_map) & np.isfinite(lower_time_map) &
    (upper_time_map <= lower_time_map) & (upper_time_map >= time_start) &
    (lower_time_map <= time_end))

print("Number of valid horizon grid points:", np.sum(valid_horizon))
print("Number of invalid horizon grid points:", np.sum(~valid_horizon))

if not np.any(valid_horizon):
    raise ValueError("No valid clipping region exists between the upper and lower boundary horizons.")

#%% Construct Time Axis and Crop the Prediction Volume Using Horizon Constraints
time_axis = time_start + np.arange(n_time) * dt

interval_mask = (
    valid_horizon[:, :, np.newaxis] &(time_axis[np.newaxis, np.newaxis, :] >=
        upper_time_map[:, :, np.newaxis]) & (time_axis[np.newaxis, np.newaxis, :] <=
        lower_time_map[:, :, np.newaxis]))

cropped_volume = ((binary_volume == 1) & interval_mask).astype(np.uint8)

print("Cropped 3D volume shape:", cropped_volume.shape)
print("Number of positive voxels in the cropped volume:", np.sum(cropped_volume == 1))

if np.sum(cropped_volume == 1) == 0:
    raise ValueError(
    "No positive voxels found between the upper and lower boundary horizons. "
    "Unable to reconstruct the 3D volume.")

#%% 7. Extract the 3D Volume Surface Using Marching Cubes
padded_volume = np.pad(cropped_volume,pad_width=1,
    mode="constant",constant_values=0)

verts, faces, normals, values = marching_cubes(padded_volume,level=0.5)

verts -= 1.0

print("Number of surface vertices:", verts.shape[0])
print("Number of triangular faces:", faces.shape[0])

#%% Convert Index Coordinates to Real-World Coordinates
inline_coords = inline_start + verts[:, 0]
cdp_coords = cdp_start + verts[:, 1]
time_coords = time_start + verts[:, 2] * dt

z_plot = -time_coords

verts_real = np.column_stack([cdp_coords, inline_coords, z_plot])

#%% Convert to a PyVista Mesh
faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32),
     faces.astype(np.int32)]).ravel()

cavern_mesh = pv.PolyData(verts_real, faces_pv)

#%% Mesh Simplification to Reduce Rendering Lag
use_decimate = False
target_reduction = 0.35

if use_decimate:
    cavern_mesh = cavern_mesh.decimate(target_reduction=target_reduction)
    print("Number of vertices after simplification:", cavern_mesh.n_points)
    print("Number of faces after simplification:", cavern_mesh.n_cells)

#%% Load Sonar Measurement Points
sonar_clouds = {}

for sonar_name, file_name in sonar_files.items():

    sonar_path = os.path.join(sonar_dir, file_name)

    sonar_df = pd.read_csv(sonar_path)
    sonar_df.columns = sonar_df.columns.str.strip()

    required_cols = ["Xline", "Inline", "Time"]
    if not all(col in sonar_df.columns for col in required_cols):
        raise ValueError(f"{file_name} is missing required columns: Xline, Inline, and Time.")

    for col in required_cols:
        sonar_df[col] = pd.to_numeric(sonar_df[col], errors="coerce")

    sonar_df = sonar_df.dropna(subset=required_cols).copy()

    sonar_df = sonar_df[(sonar_df["Inline"] >= inline_start) &
        (sonar_df["Inline"] <= inline_end) & (sonar_df["Xline"] >= cdp_start) &
        (sonar_df["Xline"] <= cdp_end) & (sonar_df["Time"] >= time_start) &
        (sonar_df["Time"] <= time_end)].copy()

    if sonar_df.empty:
        raise ValueError(f"No valid sonar points found for {sonar_name} after filtering.")

    # X = Xline, Y = Inline, Z = -Time
    sonar_points = np.column_stack([
        sonar_df["Xline"].to_numpy(dtype=float),
        sonar_df["Inline"].to_numpy(dtype=float),
        -sonar_df["Time"].to_numpy(dtype=float)])

    sonar_clouds[sonar_name] = pv.PolyData(sonar_points)

    print(f"Number of valid sonar points for {sonar_name}: {len(sonar_points)}")

print("\nSonar data loaded successfully.")


#%% Visualization Parameters
cavern_color = "#76A7CF"
cavern_opacity = 0.28

sonar_colors = {"Sonar_1": "#D98255", "Sonar_2": "#B75E81"}
sonar_point_size = 2
background_color = "white"

#%% Create PyVista Plotter
plotter = pv.Plotter(window_size=(1400, 1000))
plotter.set_background(background_color)

try:
    plotter.enable_anti_aliasing()
except:
    pass

plotter.add_mesh(
    cavern_mesh,
    color=cavern_color,
    opacity=cavern_opacity,
    smooth_shading=True,
    show_edges=False,
    specular=0.15,
    specular_power=10,
    name="Predicted salt cavern")

for sonar_name, sonar_cloud in sonar_clouds.items():
    plotter.add_mesh(
        sonar_cloud,
        color=sonar_colors[sonar_name],
        point_size=sonar_point_size,
        render_points_as_spheres=True,
        name=f"{sonar_name}_sonar")


#%% Add Legend (Optional)
legend_entries = [["Predicted Salt Cavern", cavern_color],
    ["Sonar 1", sonar_colors["Sonar_1"]],
    ["Sonar 2", sonar_colors["Sonar_2"]]]

plotter.add_legend(
    labels=legend_entries,
    bcolor="white",
    face="circle",
    border=True,
    size=(0.22, 0.18))

#%% Display Coordinate Axes
z_min = np.nanmin(upper_time_map[valid_horizon])
z_max = np.nanmax(lower_time_map[valid_horizon])

bounds = (
    cdp_start, cdp_end,
    inline_start, inline_end,
    -(z_max + 5), -(z_min - 5))

axes_ranges = (cdp_start, cdp_end,
    inline_start, inline_end,
    z_max + 5, z_min - 5)

plotter.show_bounds(
    bounds=bounds,
    axes_ranges=axes_ranges,

    xtitle="CDP",
    ytitle="Inline",
    ztitle="Time (ms)",

    color="black",
    font_size=12,
    bold=False,

    n_xlabels=4,
    n_ylabels=4,
    n_zlabels=5,

    fmt="%.0f",
    grid=False,
    location="outer",
    all_edges=False,
    ticks="outside")

#%% Set the Viewing Angle for the 3D Figure

center = np.array(cavern_mesh.center)
bounds = cavern_mesh.bounds

x_range = bounds[1] - bounds[0]
y_range = bounds[3] - bounds[2]
z_range = bounds[5] - bounds[4]

radius = 2.0 * max(x_range, y_range, z_range)

azimuth = 45
elevation = 20

az = np.deg2rad(azimuth)
el = np.deg2rad(elevation)

camera_position = (
    center[0] + radius * np.cos(el) * np.cos(az),
    center[1] - radius * np.cos(el) * np.sin(az),
    center[2] + radius * np.sin(el))

plotter.camera_position = [
    camera_position,
    center,
    (0, 0, 1)]

plotter.reset_camera_clipping_range()
plotter.camera.zoom(1.15)

#%%
plotter.show()
