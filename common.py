import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import streamlit as st

def map_cm(cm, class_dict):
    old_classes = sorted(class_dict.keys())
    mapping = torch.tensor([class_dict[i] for i in old_classes], dtype=torch.long)

    n_new = int(mapping.max().item()) + 1
    new_cm = torch.zeros((n_new, n_new), dtype=cm.dtype)

    for i_old in range(cm.shape[0]):
        for j_old in range(cm.shape[1]):
            i_new = class_dict[i_old]
            j_new = class_dict[j_old]
            new_cm[i_new, j_new] += cm[i_old, j_old]

    return new_cm

def map_df(df, mapping_dict, unified_class_dict, metric_cols):

    df = df.copy()
    df["old_id"] = df.index.astype(int)
    df["new_id"] = df["old_id"].map(mapping_dict)

    df['Pixel count'] = pd.to_numeric(df['Pixel count'], errors="coerce").fillna(0.0)

    rows = []
    for new_id, g in df.groupby("new_id", sort=True):
        w = g['Pixel count'].to_numpy(dtype=float)
        row = {
            "new_id": int(new_id),
            'Class name': unified_class_dict.get(int(new_id), f"Class {new_id}"),
            'Pixel count': int(w.sum()),
        }

        for c in metric_cols:
            x = g[c].to_numpy(dtype=float)
            mask = np.isfinite(x) & (w > 0)
            row[c] = float(np.average(x[mask], weights=w[mask])) if mask.any() else np.nan

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("new_id").drop(columns=["new_id"])
    return out

def map_volume(volume, mapping_dict):
    lut = np.zeros(max(mapping_dict.keys()) + 1, dtype=volume.dtype)
    for k, v in mapping_dict.items():
        lut[k] = v
    return lut[volume]

def normalize_cm(cm):
    cm = cm.clone().float()
    row_sums = cm.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums

def get_top_confusions(cm, top_k=20, class_dict=None):
    """
    cm: normalized confusion matrix (N x N), rows = true, cols = pred
    class_names: list of names or None
    top_k: how many confusions to return
    """
    N = cm.shape[0]

    # Copy so we don’t modify the original
    cm2 = cm.clone()

    # Zero out the diagonal (correct predictions)
    cm2[torch.eye(N, dtype=torch.bool)] = 0.0

    # Flatten to list of (value, true_class, pred_class)
    values = []
    for i in range(N):
        for j in range(N):
            if i != j:
                values.append((cm2[i, j].item(), i, j))

    # Sort descending by confusion value
    values.sort(reverse=True, key=lambda x: x[0])

    # Take top K
    results = values[:top_k]

    # Optional: convert indices to names
    if class_dict is not None:
        readable = []
        for v, i, j in results:
            readable.append({
                "true_class": f"{i}: {class_dict[i]}",
                "predicted_as": f"{j}: {class_dict[j]}",
                "value": v
            })
        return readable

    return results

def plot_cm(cm, class_dict=None, n_classes=int, title=None, pixel_counts=None, streamlit=False):
    """
    cm: (N_CLASSES, N_CLASSES) tensor
    """
    cm_np = cm.numpy()

    fig, ax = plt.subplots(figsize=(15, 15), dpi=150)
    im = ax.imshow(cm_np, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title or "Confusion matrix")

    if class_dict is not None:
        ax.set_xticks(range(len(class_dict)))
        ax.set_yticks(range(len(class_dict)))
        ax.set_xticklabels([f"{k}: {v}" for k, v in class_dict.items()], rotation=45, ha='right')
        if pixel_counts is not None:
            ax.set_yticklabels([f"{k}: {v} ({pixel_counts[k]})" for k, v in class_dict.items()])
        else:
            ax.set_yticklabels([f"{k}: {v}" for k, v in class_dict.items()])
        ax.tick_params(axis='both', which='major', labelsize=12)
    else:
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))

    plt.tight_layout()

    if streamlit:
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        plt.show()

def plot_hist(vol: np.ndarray, bins: int = 256):
    x = vol.ravel()

    plt.figure()
    plt.hist(x, bins=bins)
    plt.xlabel("Intensity")
    plt.ylabel("Voxel count")
    plt.show()

def viz_axes(volume, spacing=None, cmap='tab20'):

    sx, sy, sz = volume.shape
    x_slider = widgets.IntSlider(description='X', value=sx//2, min=0, max=sx-1, continuous_update=True)
    y_slider = widgets.IntSlider(description='Y', value=sy//2, min=0, max=sy-1, continuous_update=True)
    z_slider = widgets.IntSlider(description='Z', value=sz//2, min=0, max=sz-1, continuous_update=True)

    plt.ioff()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.4, top=0.9)  # reduce top spacing

    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    sagittal = volume[x_slider.value, :, :].T
    coronal = volume[:, y_slider.value, :].T
    axial = volume[:, :, z_slider.value].T

    if spacing:
        im_x = axs[0].imshow(
            sagittal,  # Sagittal
            cmap=cmap,
            origin='lower',
            extent=[0, spacing[1] * volume.shape[1], 0, spacing[2] * volume.shape[2]],
            aspect='equal'
        )

        im_y = axs[1].imshow(
            coronal,  # Coronal
            cmap=cmap,
            origin='lower',
            extent=[0, spacing[0] * volume.shape[0], 0, spacing[2] * volume.shape[2]],
            aspect='equal'
        )

        im_z = axs[2].imshow(
            axial,  # Axial
            cmap=cmap,
            origin='lower',
            extent=[0, spacing[0] * volume.shape[0], 0, spacing[1] * volume.shape[1]],
            aspect='equal'
        )
    else:
        im_x = axs[0].imshow(
            sagittal,
            cmap=cmap,
            origin='lower',
        )
        im_y = axs[1].imshow(
            coronal,
            cmap=cmap,
            origin='lower'
        )
        im_z = axs[2].imshow(
            axial,
            cmap=cmap,
            origin='lower',
        )

    axs[0].set_title(f"X={x_slider.value}"); axs[0].axis('off')
    axs[1].set_title(f"Y={y_slider.value}"); axs[1].axis('off')
    axs[2].set_title(f"Z={z_slider.value}"); axs[2].axis('off')

    def update_x(change):
        im_x.set_data(volume[x_slider.value, :, :].T)
        axs[0].set_title(f"X={x_slider.value}")
        fig.canvas.draw_idle()

    def update_y(change):
        im_y.set_data(volume[:, y_slider.value, :].T)
        axs[1].set_title(f"Y={y_slider.value}")
        fig.canvas.draw_idle()

    def update_z(change):
        im_z.set_data(volume[:, :, z_slider.value].T)
        axs[2].set_title(f"Z={z_slider.value}")
        fig.canvas.draw_idle()

    x_slider.observe(update_x, names='value')
    y_slider.observe(update_y, names='value')
    z_slider.observe(update_z, names='value')

    # Layout for sliders
    slider_box = widgets.HBox([x_slider, y_slider, z_slider])
    slider_box.layout = widgets.Layout(justify_content='center', margin='-30px 0 0 0')

    container = widgets.VBox([fig.canvas, slider_box], layout=widgets.Layout(align_items='center'))
    display(container)
    plt.ion()


def viz(pred=None,
             gt=None,
             template=None,
             spacing=None,
             show_overlap=True,
             show_template=False,
             n_classes=None,
             class_dict=None,
             show_bg=False,
             cmap='tab20',
             ):

    if pred is not None and not isinstance(pred, list):
        pred = [pred]
    if gt is not None and not isinstance(gt, list):
        gt = [gt]
    if template is not None and not isinstance(template, list):
        template = [template]
    if class_dict is None:
        class_dict = {i: str(i) for i in range(n_classes)}
    else:
        n_classes = len(class_dict)

    show_overlap = show_overlap and pred is not None and gt is not None
    show_template = show_template and template is not None

    ax_idx = {}
    idx = 0
    if show_template is True:
        ax_idx['template'] = idx
        idx += 1
    if pred is not None:
        ax_idx['pred'] = idx
        idx += 1
    if gt is not None:
        ax_idx['gt'] = idx
        idx += 1
    if show_overlap is True:
        ax_idx['overlap'] = idx

    if pred is not None:
        n_slices, x, y = pred[0].shape
        n_volumes = len(pred)
    elif template is not None:
        n_slices, x, y = template[0].shape
        n_volumes = len(template)
    elif gt is not None:
        n_slices, x, y = gt[0].shape
        n_volumes = len(gt)

    slider = widgets.IntSlider(
        description='Y',
        value=n_slices//2,
        min=0,
        max=n_slices-1,
        continuous_update=True,
    )
    dropdown = widgets.Dropdown(
        description='Class',
        options=[(f"Class {i}: {class_dict[i]}", i) for i in range(n_classes)] + [("No selection", None)],
        value=None,
    )
    volume_dropdown = widgets.Dropdown(
        description='Volume index',
        options=list(range(n_volumes)),
        value=0,
    )

    if not show_bg or show_template is False:
        cmap = plt.get_cmap(cmap, n_classes)
        colors = cmap(np.arange(n_classes))
        colors[0, -1] = 0.0
        cmap = ListedColormap(colors)

    plt.ioff()
    fig, axs = plt.subplots(1, len(ax_idx), figsize=(14, 4))
    axs = np.atleast_1d(axs)
    fig.subplots_adjust(wspace=0.1, top=0.9)  # reduce top spacing

    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    if show_template is True:

        template_slice = template[volume_dropdown.value][slider.value,:,  :].T

        im_template = axs[0].imshow(
            template_slice,  # Sagittal
            cmap='gray',
            origin='lower',
            extent=[0, spacing[1] * x, 0, spacing[2] * y] if spacing is not None else None,
            aspect='equal'
        )
        axs[ax_idx['template']].set_title('Template')
        axs[ax_idx['template']].axis('off')

    if show_template is False:

        template_slice = template[volume_dropdown.value][slider.value,:,  :].T
        if pred is not None:
            im_pred_bg = axs[ax_idx['pred']].imshow(
                template_slice,  # Sagittal
                cmap='gray',
                origin='lower',
                extent=[0, spacing[1] * x, 0, spacing[2] * y] if spacing is not None else None,
                aspect='equal'
            )
        if gt is not None:
            im_gt_bg = axs[ax_idx['gt']].imshow(
                template_slice,  # Sagittal
                cmap='gray',
                origin='lower',
                extent=[0, spacing[1] * x, 0, spacing[2] * y] if spacing is not None else None,
                aspect='equal'
            )

    if pred is not None:
        slice_pred = pred[volume_dropdown.value][slider.value,:,  :].T

        im_pred = axs[ax_idx['pred']].imshow(
            slice_pred,  # Sagittal
            cmap=cmap,
            vmin=0,
            vmax=n_classes-1,
            origin='lower',
            extent=[0, spacing[1] * x, 0, spacing[2] * y] if spacing is not None else None,
            aspect='equal',
            alpha=0.3 if show_template is False else 1,
        )
        axs[ax_idx['pred']].set_title('Prediction')
        axs[ax_idx['pred']].axis('off')

    if gt is not None:
        slice_gt = gt[volume_dropdown.value][slider.value,:,  :].T

        im_gt = axs[ax_idx['gt']].imshow(
            slice_gt,
            cmap=cmap,
            vmin=0,
            vmax=n_classes-1,
            origin='lower',
            extent=[0, spacing[1] * x, 0, spacing[2] * y] if spacing is not None else None,
            aspect='equal',
            alpha=0.3 if show_template is False else 1,
        )
        axs[ax_idx['gt']].set_title('Ground truth')
        axs[ax_idx['gt']].axis('off')

    if show_overlap:
        overlap = slice_pred == slice_gt

        im_overlap = axs[ax_idx['overlap']].imshow(
            overlap,
            cmap=ListedColormap(['red', 'white', 'black']),
            origin='lower',
            extent=[0, spacing[1] * x, 0, spacing[2] * y] if spacing is not None else None,
            vmin=0,
            vmax=2,
            aspect='equal'
        )
        axs[ax_idx['overlap']].set_title('Overlap')
        axs[ax_idx['overlap']].axis('off')

    fig_legend = None

    def update(change):
        filter = dropdown.value

        if pred is not None:
            pred_slice = pred[volume_dropdown.value][slider.value,:,  :].T

            if filter is None:
                im_pred.set_data(pred_slice)
                im_pred.set_cmap(cmap)
                im_pred.set_clim(0, n_classes - 1)
            else:
                im_pred.set_data(pred_slice == filter)
                im_pred.set_cmap(ListedColormap(['black', plt.get_cmap(cmap)(filter)]))
                im_pred.set_clim(0, 1)

        if show_template is True:
            template_slice = template[volume_dropdown.value][slider.value,:,  :].T
            im_template.set_data(template_slice)

        if show_template is False:
            template_slice = template[volume_dropdown.value][slider.value,:,  :].T
            if pred is not None:
                im_pred_bg.set_data(template_slice)
            if gt is not None:
                im_gt_bg.set_data(template_slice)

        if gt is not None:
            gt_slice = gt[volume_dropdown.value][slider.value,:,  :].T
            if filter is None:
                im_gt.set_data(gt_slice)
                im_gt.set_cmap(cmap)
                im_gt.set_clim(0, n_classes - 1)
            else:
                im_gt.set_data(gt_slice == filter)
                im_gt.set_cmap(ListedColormap(['black', plt.get_cmap(cmap)(filter)]))
                im_gt.set_clim(0, 1)

        if show_overlap:
            if filter is None:
                im_overlap.set_data(pred_slice == gt_slice)
            else:
                overlap_img = np.full_like(pred_slice, 2, dtype=np.uint8)
                overlap_img[(pred_slice == filter) & (gt_slice == filter)] = 1
                overlap_img[(pred_slice == filter) ^ (gt_slice == filter)] = 0
                im_overlap.set_data(overlap_img)

        nonlocal fig_legend
        if fig_legend is not None:
            fig_legend.remove()
        if filter is not None:
            return

        visible_classes = set()
        if pred:
            visible_classes = visible_classes.union(np.unique(pred_slice))
        if gt:
            visible_classes = visible_classes.union(np.unique(gt_slice))
        handles = []

        for class_id, class_name in class_dict.items():
            if visible_classes is not None and class_id not in visible_classes:
                continue

            color = cmap(class_id)

            handles.append(
                Patch(
                    facecolor=color,
                    edgecolor="black",
                    label=f"{class_id}: {class_name}",
                )
            )

        fig_legend = fig.legend(
            handles=handles,
            loc="lower center",
            fontsize=6,
            ncols=min(len(visible_classes), 6),
            frameon=False,
        )

        #fig.suptitle(f"{slider.value}")
        fig.canvas.draw_idle()

    slider.observe(update, names='value')
    dropdown.observe(update, names='value')
    volume_dropdown.observe(update, names='value')

    controls = widgets.HBox([volume_dropdown, slider, dropdown])
    container = widgets.VBox(
        [fig.canvas, controls],
        layout=widgets.Layout(align_items='center')
    )
    display(container)
    update(None)
    plt.ion()

def viz_streamlit(
    pred=None,
    gt=None,
    template=None,
    spacing=None,
    show_overlap=True,
    show_template=False,
    n_classes=None,
    class_dict=None,
    show_bg=False,
    cmap_name="tab20",
):
    """
    Streamlit version of your viz().

    Expected shapes:
      pred: (n_slices, x, y) or list of such volumes
      gt:   (n_slices, x, y) or list of such volumes
      template: (n_slices, x, y) or list of such volumes
    """

    # --- Normalize inputs to lists ---
    if pred is not None and not isinstance(pred, list):
        pred = [pred]
    if gt is not None and not isinstance(gt, list):
        gt = [gt]
    if template is not None and not isinstance(template, list):
        template = [template]

    # --- Class dict / n_classes handling ---
    if class_dict is None:
        if n_classes is None:
            raise ValueError("Provide n_classes if class_dict is None.")
        class_dict = {i: str(i) for i in range(n_classes)}
    else:
        n_classes = len(class_dict)

    show_overlap = bool(show_overlap and pred is not None and gt is not None)
    show_template = bool(show_template and template is not None)

    # --- Determine geometry & number of volumes ---
    ref = None
    if pred is not None:
        ref = pred[0]
        n_volumes = len(pred)
    elif template is not None:
        ref = template[0]
        n_volumes = len(template)
    elif gt is not None:
        ref = gt[0]
        n_volumes = len(gt)
    else:
        raise ValueError("At least one of pred/gt/template must be provided.")

    n_slices, x, y = ref.shape

    # --- Ax layout mapping ---
    ax_idx = {}
    idx = 0
    if show_template:
        ax_idx["template"] = idx
        idx += 1
    if pred is not None:
        ax_idx["pred"] = idx
        idx += 1
    if gt is not None:
        ax_idx["gt"] = idx
        idx += 1
    if show_overlap:
        ax_idx["overlap"] = idx
        idx += 1

    n_axes = len(ax_idx)
    if n_axes == 0:
        raise ValueError("Nothing to plot (check flags and provided inputs).")

    # --- Widgets (Streamlit) ---
    c1, c2, c3 = st.columns([1, 2, 2])
    with c1:
        vol_idx = st.selectbox("Volume index", list(range(n_volumes)), index=0)
    with c2:
        slice_idx = st.slider("Y", min_value=0, max_value=n_slices - 1, value=n_slices // 2, step=1)
    with c3:
        class_options = [("No selection", None)] + [(f"Class {i}: {class_dict[i]}", i) for i in range(n_classes)]
        class_label = st.selectbox("Class", options=class_options, index=0, format_func=lambda x: x[0])
        filter_class = class_label[1]

    # --- Colormap setup (match your behavior re background/alpha) ---
    # When not showing background OR template is shown as its own panel:
    # set alpha=0 for class 0 (background) by default, as in your code.
    cmap = plt.get_cmap(cmap_name, n_classes)
    if (not show_bg) or show_template:
        colors = cmap(np.arange(n_classes))
        colors[0, -1] = 0.0  # transparent background class
        cmap = ListedColormap(colors)

    # --- Extent handling ---
    extent = None
    if spacing is not None:
        # your code uses spacing[1]*x, spacing[2]*y
        extent = [0, spacing[1] * x, 0, spacing[2] * y]

    # --- Pull slices (transpose + sagittal style, matching your .T) ---
    template_slice = None
    if template is not None:
        template_slice = template[vol_idx][slice_idx, :, :].T

    pred_slice = None
    if pred is not None:
        pred_slice = pred[vol_idx][slice_idx, :, :].T

    gt_slice = None
    if gt is not None:
        gt_slice = gt[vol_idx][slice_idx, :, :].T

    # --- Create figure ---
    fig, axs = plt.subplots(1, n_axes, figsize=(15, 3))
    axs = np.atleast_1d(axs)
    fig.subplots_adjust(wspace=0.1, top=0.9)

    def _imshow(ax, img, cm, vmin=None, vmax=None, alpha=1.0):
        return ax.imshow(
            img,
            cmap=cm,
            origin="lower",
            extent=extent,
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
        )

    # --- TEMPLATE panel ---
    if show_template:
        ax = axs[ax_idx["template"]]
        _imshow(ax, template_slice, "gray")
        ax.set_title("Template")
        ax.axis("off")

    # --- Background template behind pred/gt when show_template is False ---
    if (not show_template) and (template_slice is not None):
        if pred is not None:
            ax = axs[ax_idx["pred"]]
            _imshow(ax, template_slice, "gray")
        if gt is not None:
            ax = axs[ax_idx["gt"]]
            _imshow(ax, template_slice, "gray")

    # --- PRED panel ---
    if pred is not None:
        ax = axs[ax_idx["pred"]]
        if filter_class is None:
            _imshow(
                ax,
                pred_slice,
                cmap,
                vmin=0,
                vmax=n_classes - 1,
                alpha=0.3 if (not show_template and template_slice is not None) else 1.0,
            )
        else:
            # binary mask for selected class
            mask = (pred_slice == filter_class).astype(np.uint8)
            sel_color = plt.get_cmap(cmap_name)(filter_class)
            bin_cmap = ListedColormap(["black", sel_color])
            _imshow(ax, mask, bin_cmap, vmin=0, vmax=1, alpha=1.0)
        ax.set_title("Prediction")
        ax.axis("off")

    # --- GT panel ---
    if gt is not None:
        ax = axs[ax_idx["gt"]]
        if filter_class is None:
            _imshow(
                ax,
                gt_slice,
                cmap,
                vmin=0,
                vmax=n_classes - 1,
                alpha=0.3 if (not show_template and template_slice is not None) else 1.0,
            )
        else:
            mask = (gt_slice == filter_class).astype(np.uint8)
            sel_color = plt.get_cmap(cmap_name)(filter_class)
            bin_cmap = ListedColormap(["black", sel_color])
            _imshow(ax, mask, bin_cmap, vmin=0, vmax=1, alpha=1.0)
        ax.set_title("Ground truth")
        ax.axis("off")

    # --- OVERLAP panel ---
    if show_overlap:
        ax = axs[ax_idx["overlap"]]
        if filter_class is None:
            overlap = (pred_slice == gt_slice).astype(np.uint8)
            # 0/1 mapped via a 3-color cmap in your original, but you effectively used bool
            # Here we preserve the spirit: mismatch red, match white.
            ov_cmap = ListedColormap(["red", "white"])
            _imshow(ax, overlap, ov_cmap, vmin=0, vmax=1, alpha=1.0)
        else:
            # 2=background (black), 1=both match (white), 0=exactly one matches (red)
            overlap_img = np.full_like(pred_slice, 2, dtype=np.uint8)
            overlap_img[(pred_slice == filter_class) & (gt_slice == filter_class)] = 1
            overlap_img[(pred_slice == filter_class) ^ (gt_slice == filter_class)] = 0
            ov_cmap = ListedColormap(["red", "white", "black"])
            _imshow(ax, overlap_img, ov_cmap, vmin=0, vmax=2, alpha=1.0)

        ax.set_title("Overlap")
        ax.axis("off")

    # --- Legend (only when no class is selected; match your behavior) ---
    if filter_class is None:
        visible_classes = set()
        if pred_slice is not None:
            visible_classes |= set(np.unique(pred_slice).tolist())
        if gt_slice is not None:
            visible_classes |= set(np.unique(gt_slice).tolist())

        handles = []
        for class_id, class_name in class_dict.items():
            if class_id not in visible_classes:
                continue
            color = cmap(class_id)
            handles.append(Patch(facecolor=color, edgecolor="black", label=f"{class_id}: {class_name}"))

        if handles:
            fig.legend(
                handles=handles,
                loc="lower center",
                fontsize=6,
                ncols=min(len(visible_classes), 6),
                frameon=False,
            )

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)