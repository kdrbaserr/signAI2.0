"""Report that the installed package is available."""

from signai import __version__


def main() -> None:
    print(f"signai-2.0 {__version__}: environment ready")


if __name__ == "__main__":
    main()
