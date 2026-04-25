"""
Wiki backup - local tarball creation and rotation.

Creates timestamped tarballs of the vault directory.
Keeps last N backups, deletes older ones.
"""
import logging
import tarfile
from datetime import datetime
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class WikiBackup:
    """Tarball backup manager for the wiki vault."""

    def __init__(self, vault_path: Path, backup_dir: Path, keep_last_n: int = 30) -> None:
        self.vault_path = vault_path
        self.backup_dir = backup_dir
        self.keep_last_n = keep_last_n

    def create_tarball(self) -> Path:
        """
        Create a timestamped tarball of the vault.
        Returns the path to the new tarball.
        """
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tarball_name = f"vault-{timestamp}.tar.gz"
        tarball_path = self.backup_dir / tarball_name

        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(self.vault_path, arcname="vault")

        size_mb = tarball_path.stat().st_size / (1024 * 1024)
        logger.info("Created vault backup: %s (%.1f MB)", tarball_name, size_mb)

        self._rotate_backups()
        return tarball_path

    def list_backups(self) -> List[Path]:
        """List existing backups sorted by date (newest first)."""
        if not self.backup_dir.exists():
            return []
        backups = sorted(
            self.backup_dir.glob("vault-*.tar.gz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return backups

    def _rotate_backups(self) -> None:
        """Delete backups beyond the keep_last_n limit."""
        backups = self.list_backups()
        for old_backup in backups[self.keep_last_n:]:
            old_backup.unlink()
            logger.info("Deleted old backup: %s", old_backup.name)
