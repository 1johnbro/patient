"""Multi-turn CLI for the scheduling assistant.

One Orchestrator stays alive for the whole session.
"""

import sys

from orchestrator import Orchestrator


def main(argv):
    bot = Orchestrator()
    print("Scheduling assistant. Type quit to exit.")

    if argv:
        print(bot.handle(" ".join(argv)))
        print()

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            return 0
        print(bot.handle(text))
        print()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
