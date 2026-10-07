"""Workflow checks against the supplied mock scheduling API.

Language understanding is scripted so each run is the same. Provider, patient,
availability, booking, and handoff data still come from http://localhost:4010.

    python3 mock-api/server.py
    python3 evals.py
"""

import sys

import orchestrator
from orchestrator import Orchestrator

CONFLICT_SLOT = "slot_conflict_001"


def understood(**kwargs):
    result = {
        "intent": "unclear",
        "specialty": None,
        "location": None,
        "phone": None,
        "dob": None,
        "zip_code": None,
        "slot_number": None,
        "confirmed": None,
    }
    result.update(kwargs)
    return result


class Script:
    """Return a prepared understanding for an exact user message."""

    def __init__(self):
        self.replies = {}

    def add(self, text, result):
        self.replies[text] = result

    def __call__(self, text, model=None):
        if text not in self.replies:
            raise AssertionError("no scripted understanding for %r" % text)
        return self.replies[text]


class BookingLog:
    """Count real booking calls without replacing the API."""

    def __init__(self):
        self.calls = []
        self._real = orchestrator.book_appointment

        def wrapped(patient_id, slot_id, confirmed, **kwargs):
            self.calls.append((patient_id, slot_id, confirmed))
            return self._real(patient_id, slot_id, confirmed, **kwargs)

        orchestrator.book_appointment = wrapped

    def restore(self):
        orchestrator.book_appointment = self._real


def provider_lookup():
    script = Script()
    script.add(
        "Which primary care providers are downtown?",
        understood(intent="provider_lookup", specialty="primary_care", location="downtown"),
    )
    reply = Orchestrator(script).handle("Which primary care providers are downtown?")
    if "Providers:" not in reply or "Dr. Elena Brooks" not in reply or "Dr. Marcus King" not in reply:
        raise AssertionError(reply)


def happy_path_booking():
    bot, script = _booking_bot("555-0101", "1985-04-12", "primary_care", "downtown")
    listed = _through_availability(bot, script, "555-0101", "1985-04-12")
    if "Available times:" not in listed or not bot.slots:
        raise AssertionError(listed)

    bookable = [slot for slot in bot.slots if slot.get("slotId") != CONFLICT_SLOT]
    if not bookable:
        raise AssertionError("no bookable downtown primary care slot left; restart the mock API")
    chosen = bookable[0]
    number = bot.slots.index(chosen) + 1

    log = BookingLog()
    try:
        script.add(str(number), understood(slot_number=number))
        asked = bot.handle(str(number))
        if log.calls:
            raise AssertionError("booked before confirmation: %s" % (log.calls,))
        if "Say yes to book it" not in asked or "Your appointment is" in asked:
            raise AssertionError(asked)
        if bot.selected_slot is not chosen:
            raise AssertionError("selected slot was not the API slot")

        script.add("yes", understood())
        booked = bot.handle("yes")
        if log.calls != [(bot.patient["patientId"], chosen["slotId"], True)]:
            raise AssertionError(log.calls)
        if not booked.startswith("Your appointment is scheduled"):
            raise AssertionError(booked)
    finally:
        log.restore()


def no_patient():
    bot, script = _booking_bot("555-9999", "1990-01-01", "primary_care", "downtown")
    log = BookingLog()
    try:
        reply = _through_availability(bot, script, "555-9999", "1990-01-01")
        if log.calls or bot.slots or bot.patient is not None:
            raise AssertionError("continued after no patient match")
        if "Available times:" in reply or "Your appointment is" in reply:
            raise AssertionError(reply)
        if "couldn't find a patient" not in reply or "won't guess" not in reply:
            raise AssertionError(reply)
        if "queued this for a staff member" not in reply:
            raise AssertionError(reply)
    finally:
        log.restore()


def no_availability():
    bot, script = _booking_bot("555-0102", "1992-06-03", "dermatology", "lakeside")
    log = BookingLog()
    try:
        reply = _through_availability(bot, script, "555-0102", "1992-06-03")
        if log.calls or bot.slots:
            raise AssertionError("offered or booked a slot the API did not return")
        if "Nothing is available for dermatology at lakeside." not in reply:
            raise AssertionError(reply)
        if "Your appointment is" in reply:
            raise AssertionError(reply)
    finally:
        log.restore()


def slot_conflict():
    bot, script = _booking_bot("555-0101", "1985-04-12", "primary_care", "downtown")
    listed = _through_availability(bot, script, "555-0101", "1985-04-12")
    numbers = [i for i, slot in enumerate(bot.slots, start=1) if slot.get("slotId") == CONFLICT_SLOT]
    if len(numbers) != 1:
        raise AssertionError("slot_conflict_001 was not selectable: %s" % listed)
    number = numbers[0]

    log = BookingLog()
    try:
        script.add(str(number), understood(slot_number=number))
        asked = bot.handle(str(number))
        if log.calls or "Say yes to book it" not in asked:
            raise AssertionError(asked)
        if bot.selected_slot is None or bot.selected_slot.get("slotId") != CONFLICT_SLOT:
            raise AssertionError("conflict slot was not the selected API slot")

        script.add("yes", understood())
        failed = bot.handle("yes")
        if bot.selected_slot is not None:
            raise AssertionError("selected slot was not cleared")
        if "just taken" not in failed or "did not book" not in failed:
            raise AssertionError(failed)
        if "Your appointment is" in failed:
            raise AssertionError(failed)
    finally:
        log.restore()


def _booking_bot(phone, dob, specialty, location):
    script = Script()
    script.add(
        "I want to book an appointment",
        understood(intent="book_appointment", specialty=specialty, location=location),
    )
    script.add(phone, understood(phone=phone))
    script.add(dob, understood(dob=dob))
    return Orchestrator(script), script


def _through_availability(bot, script, phone, dob):
    bot.handle("I want to book an appointment")
    bot.handle(phone)
    return bot.handle(dob)


def main():
    checks = [
        ("provider_lookup", provider_lookup),
        ("happy_path_booking", happy_path_booking),
        ("no_patient", no_patient),
        ("no_availability", no_availability),
        ("slot_conflict", slot_conflict),
    ]
    passed = 0
    for name, check in checks:
        try:
            check()
        except Exception as exc:
            print("FAIL %s" % name)
            print("  %s" % exc)
            continue
        print("PASS %s" % name)
        passed += 1
    print("%s/%s passed" % (passed, len(checks)))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
