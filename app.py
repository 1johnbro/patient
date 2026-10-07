"""Minimal CLI for provider lookup.

Input is matched against the scheduling API's specialty and location enums,
then sent through tools.lookup_providers. No model calls.
"""

import sys

from tools import SchedulingAPIError, lookup_providers

SPECIALTIES = (
    ("primary care", "primary_care"),
    ("dermatology", "dermatology"),
)
LOCATIONS = ("downtown", "uptown", "lakeside")


def parse_provider_query(text):
    """Pull specialty and location out of a short phrase.

    Returns (specialty, location). Either may be None when that word is absent.
    """
    normalized = " ".join(text.lower().replace("_", " ").split())
    words = set(normalized.split())

    specialty = None
    for phrase, value in SPECIALTIES:
        if phrase in normalized:
            specialty = value
            break

    location = next((name for name in LOCATIONS if name in words), None)
    return specialty, location


def format_providers(providers):
    if not providers:
        return "No providers matched."

    lines = []
    for provider in providers:
        name = provider.get("name", "Unknown provider")
        specialty = str(provider.get("specialty", "")).replace("_", " ")
        locations = ", ".join(provider.get("locations") or [])
        modalities = ", ".join(provider.get("modalities") or []).replace("_", " ")
        lines.append(
            "- {} — {} — {} — {}".format(name, specialty, locations, modalities)
        )
    return "\n".join(lines)


def run_provider_lookup(text):
    specialty, location = parse_provider_query(text)
    if specialty is None and location is None:
        return (
            "Say a specialty and location, for example: primary care downtown\n"
            "Specialties: primary care, dermatology\n"
            "Locations: downtown, uptown, lakeside"
        )

    try:
        payload = lookup_providers(specialty=specialty, location=location)
    except SchedulingAPIError as exc:
        return str(exc)

    providers = payload.get("providers") or []
    filters = []
    if specialty:
        filters.append(specialty.replace("_", " "))
    if location:
        filters.append(location)
    heading = "Providers ({})".format(", ".join(filters)) if filters else "Providers"
    return "{}\n{}".format(heading, format_providers(providers))


def main(argv):
    text = " ".join(argv).strip()
    if not text:
        try:
            text = input("Provider lookup> ").strip()
        except EOFError:
            return 0
    if not text or text.lower() in {"quit", "exit"}:
        return 0

    print(run_provider_lookup(text))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
