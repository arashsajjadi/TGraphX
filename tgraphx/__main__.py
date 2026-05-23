"""Entry point for ``python -m tgraphx``.

Runs the TGraphX doctor check and prints system status.

Console scripts registered in ``pyproject.toml``:

- ``tgraphx-doctor`` — run the installation health check
- ``tgraphx-info``   — alias for ``tgraphx-doctor`` (same behaviour)

Both scripts delegate to ``main()`` in this module.

Usage::

    python -m tgraphx              # same as doctor
    python -m tgraphx doctor
    python -m tgraphx info         # alias for doctor
    python -m tgraphx capabilities
    python -m tgraphx tasks
    python -m tgraphx models
    python -m tgraphx samplers
"""
import sys


def main() -> None:
    args = sys.argv[1:]
    command = args[0] if args else "doctor"

    if command in ("doctor", "info", ""):
        from tgraphx.easy import doctor
        doctor()
    elif command == "capabilities":
        from tgraphx.easy import show_capabilities
        show_capabilities()
    elif command == "tasks":
        from tgraphx.easy import list_tasks
        tasks = list_tasks()
        print("Available TGraphX tasks:")
        for name, desc in tasks.items():
            print(f"  {name}: {desc}")
    elif command == "models":
        from tgraphx.easy import list_models
        models = list_models()
        print("Available TGraphX models:")
        for name, desc in models.items():
            print(f"  {name}: {desc}")
    elif command == "samplers":
        from tgraphx.easy import list_samplers
        samplers = list_samplers()
        print("Available TGraphX samplers:")
        for name, desc in samplers.items():
            print(f"  {name}: {desc}")
    elif command in ("readiness", "check", "audit"):
        import json as _json
        from tgraphx.ux import audit_package_readiness
        report = audit_package_readiness()
        print("TGraphX Package Readiness Report")
        print("=" * 40)
        print(f"Version: {report['tgraphx_version']}")
        print(f"Torch: {report['torch_version']}")
        print(f"CUDA: {report['cuda_available']} (devices: {report['cuda_device_count']})")
        required = report.get("required_dependencies")
        if required:
            print("\nRequired dependencies:")
            for pkg, ver in required.items():
                print(f"  {pkg}: {ver}")
        print("\nOptional dependencies:")
        for pkg, ver in report.get("optional_dependencies", {}).items():
            print(f"  {pkg}: {ver}")
        print("\nPublic API:", report.get("public_api", {}))
        print("\nWorkflow tasks:", report.get("workflow_tasks", []))
        print("\nFeatures:", {k: v for k, v in report.get("features", {}).items() if v})
        print("\nKnown limitations:")
        for lim in report.get("known_limitations", []):
            print(f"  • {lim}")
    elif command in ("list-datasets", "list_datasets"):
        from tgraphx.datasets import list_dataset_aliases, list_datasets
        print("User-friendly dataset aliases:")
        for alias, canonical in sorted(list_dataset_aliases().items()):
            print(f"  {alias:25s} → {canonical}")
    elif command in ("list-methods", "list_methods", "list-generation"):
        from tgraphx.generation import list_graph_generation_methods
        print("Graph generation methods:")
        for name, info in list_graph_generation_methods().items():
            print(f"  {name:25s} [{info.get('stability', '?')}] {info.get('description', '')}")
    elif command in ("help", "--help", "-h"):
        print(
            "Usage: python -m tgraphx [command]\n\n"
            "Commands:\n"
            "  doctor          Check TGraphX installation (default)\n"
            "  info            Alias for doctor\n"
            "  readiness       Full package readiness audit (v1.4.1+)\n"
            "  capabilities    Show all TGraphX capabilities\n"
            "  tasks           List available workflow tasks\n"
            "  models          List available models\n"
            "  samplers        List available samplers\n"
            "  list-datasets   List user-friendly dataset aliases\n"
            "  list-methods    List graph generation methods\n"
            "  help            Show this help message\n"
        )
    else:
        print(f"Unknown command: {command!r}.  Run 'python -m tgraphx help' for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
