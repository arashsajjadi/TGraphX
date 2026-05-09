"""Entry point for ``python -m tgraphx``.

Runs the TGraphX doctor check and prints system status.

Usage::

    python -m tgraphx
    python -m tgraphx doctor
    python -m tgraphx info
    python -m tgraphx capabilities
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
    elif command in ("help", "--help", "-h"):
        print(
            "Usage: python -m tgraphx [command]\n\n"
            "Commands:\n"
            "  doctor       Check TGraphX installation (default)\n"
            "  info         Alias for doctor\n"
            "  capabilities Show all TGraphX capabilities\n"
            "  tasks        List available tasks\n"
            "  models       List available models\n"
            "  samplers     List available samplers\n"
            "  help         Show this help message\n"
        )
    else:
        print(f"Unknown command: {command!r}.  Run 'python -m tgraphx help' for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
