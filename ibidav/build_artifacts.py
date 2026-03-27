from __future__ import annotations

from ibidav.service import service


def main() -> None:
    bundle = service.build_runtime_bundle()
    artifact_path = service.save_runtime_bundle(bundle)
    print(f"Saved IBiDAV artifacts to {artifact_path}")


if __name__ == "__main__":
    main()
