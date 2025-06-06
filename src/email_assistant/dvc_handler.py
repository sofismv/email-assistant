import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class DVCHandler:
    def __init__(
        self, project_root: Path, remote_path: str, remote_name: str = "local"
    ):
        """Initialize the DVCHandler.

        Args:
            project_root (Path): Root directory where DVC will be initialized
            remote_path (str): Path to the remote storage location
            remote_name (str, optional): Name for the DVC remote. Defaults to "local".
        """
        self.project_root = Path(project_root).resolve()
        self.remote_name = remote_name
        self.remote_path = Path(remote_path)

        self._init_dvc()
        self._setup_remote()

    def _init_dvc(self):
        """Initialize DVC in the project directory."""
        dvc_dir = self.project_root / ".dvc"
        if not dvc_dir.exists():
            logger.info("Initializing DVC...")
            subprocess.run(
                ["dvc", "init", "--no-scm"], cwd=self.project_root, check=True
            )
        else:
            logger.info("DVC already initialized.")

    def _setup_remote(self):
        """Set up DVC remote storage."""
        self.remote_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "dvc",
                "remote",
                "add",
                "-f",
                "-d",
                self.remote_name,
                str(self.remote_path),
            ],
            cwd=self.project_root,
            check=True,
        )
        logger.info(f"DVC remote '{self.remote_name}' added at: {self.remote_path}")

    def add_and_push(self, path: Path):
        """Add a file to DVC tracking and push to remote storage.

        Args:
            path (Path): Path to the file to be added and pushed
        """
        subprocess.run(["dvc", "add", str(path)], cwd=self.project_root, check=True)
        subprocess.run(["dvc", "push"], cwd=self.project_root, check=True)

    def pull(self, path: Path):
        """Pull a specific file from DVC remote storage.

        Args:
            path (Path): Path to the file to be pulled from remote storage
        """
        subprocess.run(["dvc", "pull", str(path)], cwd=self.project_root, check=True)

    def exists_in_dvc(self, path: Path) -> bool:
        """Determines if a file is under DVC version control.

        Args:
            path (Path): Path to the file to check

        Returns:
            bool: True if the file is tracked by DVC, False otherwise
        """
        dvc_file = self.project_root / f"{path}.dvc"
        return dvc_file.exists()

    def exists_in_data(self, path: Path) -> bool:
        """Check if a file exists in the local filesystem.

        Args:
            path (Path): Path to the file to check

        Returns:
            bool: True if the file exists locally, False otherwise
        """
        return path.exists()
