"""CLI entry point for ROCm benchmark tool."""

from .app import app


def main() -> None:
    """Run the ROCm benchmark CLI application."""
    app()


if __name__ == "__main__":
    main()
