import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import ezdxf
import hdbscan
import numpy as np
import pandera as pa
import polars as pl
import wx
from joblib import Parallel, delayed
from pandera import check_types
from pandera.typing import Series
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import model_validator
from skimage.measure import CircleModel, ransac
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import sys

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


class Circle(BaseModel):
    """Pydantic model for circle representation."""
    xc: float
    yc: float
    r: float


class Settings(BaseModel):
    """Pydantic model for application settings."""
    input_file: Path = Field(
        ..., description="Path to the input vertex file."
    )
    output_file: Path = Field(..., description="Path to the output DXF file.")
    min_cluster_size: int = Field(
        5, ge=1, description="Minimum cluster size for HDBSCAN."
    )
    min_radius: float = Field(
        ..., gt=0.0, description="Minimum circle radius."
    )
    max_radius: float = Field(
        ..., gt=0.0, description="Maximum circle radius."
    )
    adaptive_threshold: bool = Field(
        True, description="Use adaptive threshold for RANSAC."
    )
    n_jobs: int = Field(
        default_factory=os.cpu_count,
        description="Number of parallel jobs for cluster processing.",
    )

    @field_validator("max_radius")
    def max_radius_must_be_greater_than_min_radius(cls, v, info):
        """Ensure max_radius exceeds min_radius."""
        min_radius = info.data.get("min_radius")
        if min_radius is not None and v <= min_radius:
            raise ValueError("max_radius must be greater than min_radius")
        return v

    @model_validator(mode="after")
    def check_total_radius_consistency(self) -> "Settings":
        """Validate radius consistency after all fields are set."""
        if self.max_radius <= self.min_radius:
            raise ValueError("max_radius must exceed min_radius")
        return self

    @field_validator("input_file")
    def input_file_exists(cls, v, info):
        """Check if input file exists."""
        if not v.exists():
            raise ValueError(f"Input file does not exist: {v}")
        return v

    model_config = ConfigDict(validate_assignment=True)


class VertexDataSchema(pa.DataFrameModel):
    """Pandera schema for input vertex data."""
    x: Series[np.float32] = pa.Field()
    y: Series[np.float32] = pa.Field()


class HDBSCANClusterer:
    """HDBSCAN clustering utility."""
    def __init__(self, min_cluster_size: int, min_samples: int = 1):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def cluster(self, data: np.ndarray) -> np.ndarray:
        """Cluster data using HDBSCAN.

        Args:
            data: Input data array.

        Returns:
            Array of cluster labels.
        """
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method="leaf",
        )
        clusterer.fit(data)
        return clusterer.labels_


class DXFFileSaver:
    """Utility to save circles to DXF files."""
    def save(self, circles: List[Circle], output_path: Path) -> None:
        """Save circles to a DXF file.

        Args:
            circles: List of Circle objects.
            output_path: Path to save the DXF file.
        """
        doc = ezdxf.new()
        msp = doc.modelspace()
        for circle in circles:
            if circle is not None:
                msp.add_circle(
                    center=(circle.xc, circle.yc),
                    radius=circle.r,
                )
        try:
            doc.saveas(output_path)
            logging.info(f"DXF file saved as {output_path}")
        except Exception as e:
            logging.error(f"Error saving DXF file: {e}")
            raise


@check_types
def read_vertex_file(file_path: Path) -> VertexDataSchema:
    """Read vertex data from a file.

    Args:
        file_path: Path to the vertex file.

    Returns:
        Polars DataFrame with validated data (x, y columns only).

    Raises:
        FileNotFoundError: If the file does not exist.
        pa.errors.SchemaError: If data validation fails.
        Exception: For other unexpected errors.
    """
    try:
        # Read all columns, then select and cast x, y
        df = pl.read_csv(
            str(file_path),
            has_header=False,
            new_columns=["x", "y", "z"],  # Assume 3 columns
            separator=" ",
        )
        df = df.select([
            pl.col("x").cast(pl.Float32),
            pl.col("y").cast(pl.Float32),
        ])
        logging.info(f"Data read from file: {df.shape}, dtypes: {df.dtypes}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except pa.errors.SchemaError as e:
        logging.error(f"Schema validation failed for {file_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error reading {file_path}: {e}")
        raise


def _fit_single_ransac_trial(
    points: np.ndarray,
    threshold: float,
) -> Tuple[Optional[CircleModel], np.ndarray]:
    """Perform a single RANSAC trial for circle fitting.

    Args:
        points: Array of points to fit.
        threshold: Residual threshold for RANSAC.

    Returns:
        Tuple of fitted CircleModel (or None) and inliers array.
    """
    try:
        model, inliers = ransac(
            points,
            CircleModel,
            min_samples=3,
            residual_threshold=threshold,
            max_trials=1,
        )
        return model, inliers
    except Exception as e:
        logging.error(f"Error during RANSAC fit: {e}")
        return None, np.array([])


def _validate_circle(
    model: Optional[CircleModel],
    inliers: np.ndarray,
    min_radius: float,
    max_radius: float,
    best_circle: Optional[Circle],
    best_score: float,
) -> Tuple[Optional[Circle], float]:
    """Validate a circle and update best circle if improved.

    Args:
        model: Fitted CircleModel or None.
        inliers: Array of inlier flags.
        min_radius: Minimum acceptable radius.
        max_radius: Maximum acceptable radius.
        best_circle: Current best Circle object or None.
        best_score: Current best score.

    Returns:
        Tuple of best Circle (or None) and updated score.
    """
    if model:
        xc, yc, r = model.params
        if min_radius <= r <= max_radius and inliers.sum() > 0:
            score = -inliers.sum()  # Lower is better
            if score < best_score:
                return Circle(xc=xc, yc=yc, r=r), score
    return best_circle, best_score


def fit_circle_with_magsac_characteristics(
    points: np.ndarray,
    min_radius: float,
    max_radius: float,
    adaptive_threshold: bool = True,
) -> Optional[Circle]:
    """Fit a circle using RANSAC with MAGSAC-like characteristics.

    Args:
        points: Input points array.
        min_radius: Minimum circle radius.
        max_radius: Maximum circle radius.
        adaptive_threshold: Whether to use adaptive thresholding.

    Returns:
        Fitted Circle object or None if fitting fails.
    """
    if points.shape[0] < 3:
        return None

    INITIAL_THRESHOLD = 1.0
    MAX_TRIALS = 100
    best_circle: Optional[Circle] = None
    best_score = float("inf")

    for i in tqdm(range(MAX_TRIALS), desc="Fitting Circles", unit="trial"):
        threshold = (
            INITIAL_THRESHOLD / (i + 1)
            if adaptive_threshold
            else INITIAL_THRESHOLD
        )
        model, inliers = _fit_single_ransac_trial(points, threshold)
        best_circle, best_score = _validate_circle(
            model,
            inliers,
            min_radius,
            max_radius,
            best_circle,
            best_score,
        )
    return best_circle


class CircleFitter:
    """Utility to fit circles using RANSAC."""
    def __init__(
        self,
        min_radius: float,
        max_radius: float,
        adaptive_threshold: bool = True,
    ):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.adaptive_threshold = adaptive_threshold

    def fit(self, points: np.ndarray) -> Optional[Circle]:
        """Fit a circle to the given points.

        Args:
            points: Input points array.

        Returns:
            Fitted Circle object or None.
        """
        return fit_circle_with_magsac_characteristics(
            points,
            self.min_radius,
            self.max_radius,
            self.adaptive_threshold,
        )


def process_cluster(
    data: np.ndarray,
    labels: np.ndarray,
    label: int,
    fitter: CircleFitter,
) -> Optional[Circle]:
    """Process a single cluster to fit a circle.

    Args:
        data: Full dataset array.
        labels: Cluster labels array.
        label: Specific cluster label to process.
        fitter: CircleFitter instance.

    Returns:
        Fitted Circle object or None.
    """
    points = data[labels == label]
    if points.size == 0:
        return None
    return fitter.fit(points)


def cluster_and_fit_circles(
    data: np.ndarray,
    labels: np.ndarray,
    min_radius: float,
    max_radius: float,
    n_jobs: int,
) -> List[Optional[Circle]]:
    """Cluster data and fit circles to each cluster.

    Args:
        data: Input data array.
        labels: Cluster labels array.
        min_radius: Minimum circle radius.
        max_radius: Maximum circle radius.
        n_jobs: Number of parallel jobs.

    Returns:
        List of Circle objects or None for each cluster.
    """
    fitter = CircleFitter(min_radius, max_radius)
    unique_labels = np.unique(labels)
    valid_clusters = [label for label in unique_labels if label != -1]
    n_jobs = min(n_jobs, len(valid_clusters)) if valid_clusters else 1
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cluster)(data, labels, label, fitter)
        for label in valid_clusters
    )
    return results


def setup_environment() -> None:
    """Configure environment for multi-threading."""
    n_cpus = os.cpu_count() or 1  # Fallback to 1 if None
    os.environ.update({
        k: str(n_cpus)
        for k in ("MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS")
    })


class CircleFittingApp(wx.Frame):
    """Main GUI application for circle fitting."""
    INPUT_WIDTH = 150
    TITLE_FONT_SIZE = 14
    BUTTON_BG_COLOR_RUN = "#4CAF50"
    BUTTON_BG_COLOR_EXIT = "#FF5252"
    BUTTON_FG_COLOR = wx.WHITE
    SECTION_BG_COLOR = "#f4f6f9"
    SECTION_FG_COLOR = "#4a4a4a"
    BUTTON_BG_COLOR_BROWSE = "#2196F3"
    WINDOW_SIZE = (700, 500)

    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=self.WINDOW_SIZE)
        self.settings: Optional[Settings] = None
        panel = wx.Panel(self)
        panel.SetBackgroundColour(self.SECTION_BG_COLOR)

        title_label = wx.StaticText(
            panel,
            label="Circle Fitting and Clustering Tool",
        )
        title_font = wx.Font(
            self.TITLE_FONT_SIZE,
            wx.FONTFAMILY_SWISS,
            wx.FONTSTYLE_NORMAL,
            wx.FONTWEIGHT_BOLD,
        )
        title_label.SetFont(title_font)
        title_label.SetForegroundColour(self.SECTION_FG_COLOR)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(title_label, flag=wx.ALIGN_CENTER | wx.TOP, border=20)

        input_section = wx.StaticBox(panel, label="Input Settings")
        input_section.SetForegroundColour(self.SECTION_FG_COLOR)
        input_sizer = wx.StaticBoxSizer(input_section, wx.VERTICAL)

        self.input_file_ctrl = self._add_input_row(
            panel,
            input_sizer,
            "Vertex File (.xyz):",
            "Browse",
            self._on_browse_open,
        )
        self.output_file_ctrl = self._add_input_row(
            panel,
            input_sizer,
            "Output File (.dxf):",
            "Browse",
            self._on_browse_save,
        )
        self.min_cluster_size_ctrl = self._add_input_row(
            panel,
            input_sizer,
            "Minimum Cluster Size:",
        )
        self.min_radius_ctrl = self._add_input_row(
            panel,
            input_sizer,
            "Minimum Radius:",
        )
        self.max_radius_ctrl = self._add_input_row(
            panel,
            input_sizer,
            "Maximum Radius:",
        )

        vbox.Add(input_sizer, flag=wx.ALIGN_CENTER | wx.ALL, border=10)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        run_btn = wx.Button(panel, label="Run Process")
        run_btn.SetBackgroundColour(self.BUTTON_BG_COLOR_RUN)
        run_btn.SetForegroundColour(self.BUTTON_FG_COLOR)
        run_btn.Bind(wx.EVT_BUTTON, self._on_run)

        exit_btn = wx.Button(panel, label="Exit")
        exit_btn.SetBackgroundColour(self.BUTTON_BG_COLOR_EXIT)
        exit_btn.SetForegroundColour(self.BUTTON_FG_COLOR)
        exit_btn.Bind(wx.EVT_BUTTON, self._on_exit)

        button_sizer.Add(run_btn, flag=wx.ALL, border=10)
        button_sizer.Add(exit_btn, flag=wx.ALL, border=10)

        vbox.Add(button_sizer, flag=wx.ALIGN_CENTER | wx.BOTTOM, border=20)
        panel.SetSizer(vbox)
        self.Centre()
        self.Show()

    def _add_input_row(
        self,
        panel: wx.Panel,
        sizer: wx.Sizer,
        label: str,
        button_label: str = None,
        button_callback: Optional[Callable] = None,
    ) -> wx.TextCtrl:
        """Add an input row to the GUI.

        Args:
            panel: Parent panel.
            sizer: Sizer to add the row to.
            label: Label text.
            button_label: Button text if applicable.
            button_callback: Callback for button if applicable.

        Returns:
            Text control for the input field.
        """
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        lbl = wx.StaticText(panel, label=label, size=(self.INPUT_WIDTH, -1))
        lbl.SetForegroundColour(self.SECTION_FG_COLOR)
        hbox.Add(lbl, flag=wx.RIGHT, border=8)

        text_ctrl = wx.TextCtrl(panel)
        hbox.Add(text_ctrl, proportion=1)

        if button_label and button_callback:
            btn = wx.Button(panel, label=button_label)
            btn.SetBackgroundColour(self.BUTTON_BG_COLOR_BROWSE)
            btn.SetForegroundColour(self.BUTTON_FG_COLOR)
            btn.Bind(wx.EVT_BUTTON, lambda event: button_callback(event, text_ctrl))
            hbox.Add(btn, flag=wx.LEFT, border=8)

        sizer.Add(hbox, flag=wx.EXPAND | wx.ALL, border=5)
        return text_ctrl

    def _on_browse_open(self, event: wx.Event, text_ctrl: wx.TextCtrl) -> None:
        """Handle browsing for an input file."""
        self._open_file_dialog(
            text_ctrl,
            "Open File",
            "XYZ files (*.xyz)|*.xyz",
            wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )

    def _on_browse_save(self, event: wx.Event, text_ctrl: wx.TextCtrl) -> None:
        """Handle browsing for an output file."""
        self._open_file_dialog(
            text_ctrl,
            "Save File",
            "DXF files (*.dxf)|*.dxf",
            wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )

    def _open_file_dialog(
        self,
        text_ctrl: wx.TextCtrl,
        title: str,
        wildcard: str,
        style: int,
    ) -> None:
        """Open a file dialog and set the selected path.

        Args:
            text_ctrl: Text control to update.
            title: Dialog title.
            wildcard: File type filter.
            style: Dialog style flags.
        """
        with wx.FileDialog(self, title, wildcard=wildcard, style=style) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            path = Path(dlg.GetPath())
            text_ctrl.SetValue(str(path))

    def _get_user_input(self) -> Optional[Settings]:
        """Get and validate user input from GUI.

        Returns:
            Settings object or None if validation fails.
        """
        try:
            settings = Settings(
                input_file=Path(self.input_file_ctrl.GetValue()),
                output_file=Path(self.output_file_ctrl.GetValue()),
                min_cluster_size=int(self.min_cluster_size_ctrl.GetValue()),
                min_radius=float(self.min_radius_ctrl.GetValue()),
                max_radius=float(self.max_radius_ctrl.GetValue()),
            )
            self.settings = settings
            return settings
        except ValueError as e:
            wx.MessageBox(f"Invalid input: {e}", "Error", wx.ICON_ERROR)
            return None
        except Exception as e:
            wx.MessageBox(f"Configuration error: {e}", "Error", wx.ICON_ERROR)
            return None

    def _load_data(self, input_file: Path) -> Optional[pl.DataFrame]:
        """Load data from the input file.

        Args:
            input_file: Path to the input file.

        Returns:
            DataFrame or None if loading fails.
        """
        try:
            return read_vertex_file(input_file)
        except Exception as e:
            wx.MessageBox(f"Failed to load data: {e}", "Error", wx.ICON_ERROR)
            return None

    def _cluster_data(
        self,
        data: np.ndarray,
        min_cluster_size: int,
    ) -> Optional[np.ndarray]:
        """Cluster the data using HDBSCAN.

        Args:
            data: Input data array.
            min_cluster_size: Minimum cluster size.

        Returns:
            Cluster labels or None if clustering fails.
        """
        try:
            clusterer = HDBSCANClusterer(min_cluster_size)
            labels = clusterer.cluster(data)
            silhouette_avg = silhouette_score(data, labels)
            logging.info(f"Silhouette Score: {silhouette_avg}")
            return labels
        except Exception as e:
            wx.MessageBox(f"Clustering failed: {e}", "Error", wx.ICON_ERROR)
            return None

    def _fit_circles(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        settings: Settings,
    ) -> List[Optional[Circle]]:
        """Fit circles to clustered data.

        Args:
            data: Input data array.
            labels: Cluster labels array.
            settings: Application settings.

        Returns:
            List of Circle objects or None for each cluster.
        """
        try:
            return cluster_and_fit_circles(
                data,
                labels,
                settings.min_radius,
                settings.max_radius,
                settings.n_jobs,
            )
        except Exception as e:
            wx.MessageBox(f"Circle fitting failed: {e}", "Error", wx.ICON_ERROR)
            return []

    def _save_results(
        self,
        circles: List[Circle],
        output_file: Path,
    ) -> None:
        """Save fitted circles to a DXF file.

        Args:
            circles: List of Circle objects.
            output_file: Path to save the DXF file.
        """
        try:
            saver = DXFFileSaver()
            saver.save(circles, output_file)
            wx.MessageBox(
                "Process completed successfully!",
                "Success",
                wx.ICON_INFORMATION,
            )
        except Exception as e:
            wx.MessageBox(f"Failed to save results: {e}", "Error", wx.ICON_ERROR)

    def _on_run(self, event: wx.Event) -> None:
        """Handle the Run button click."""
        settings = self._get_user_input()
        if settings is None:
            return
        logging.info(f"Running process with settings: {settings}")
        df = self._load_data(settings.input_file)
        if df is None:
            return
        data = df.to_numpy()
        labels = self._cluster_data(data, settings.min_cluster_size)
        if labels is None:
            return
        circles = self._fit_circles(data, labels, settings)
        if circles:
            self._save_results(circles, settings.output_file)

    def _on_exit(self, event: wx.Event) -> None:
        """Handle the Exit button click."""
        self.Close()


if __name__ == "__main__":
    print(sys.version)
    setup_environment()
    app = wx.App()
    CircleFittingApp(None, title="Circle Fitting and Clustering Tool")
    app.MainLoop()