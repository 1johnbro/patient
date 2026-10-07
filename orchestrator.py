"""Workflow decisions for scheduling.

Keeps the collected request, calls llm.understand for each message, and calls
tools only with values the user stated or the scheduling API returned.
Patient ids and slot ids never come from the language model.
"""

import json
import logging
import time

from llm import UnderstandingError, understand

logger = logging.getLogger("scheduling.workflow")
from tools import (
    SchedulingAPIError,
    book_appointment,
    create_handoff,
    lookup_appointments,
    lookup_availability,
    lookup_providers,
    search_patients,
)

WORKFLOW_INTENTS = ("provider_lookup", "book_appointment", "appointment_lookup")
SPECIAL_INTENTS = {
    "human_request": (
        "user_requested",
        "The user asked to speak with a person.",
        "I can't keep going on my own.",
    ),
    "medical_advice": (
        "medical_advice",
        "The user asked for medical advice.",
        "I can't give medical advice.",
    ),
    "unsupported": (
        "unsupported_request",
        "The request is outside scheduling support.",
        "I can look up providers, book an appointment, or look up an existing appointment.",
    ),
}


class Orchestrator:
    def __init__(self, understand_fn=None):
        self._understand = understand_fn or understand
        self.intent = None
        self.specialty = None
        self.location = None
        self.phone = None
        self.dob = None
        self.zip_code = None
        self.patient = None
        self.candidates = None
        self.slots = []
        self.slots_shown = False
        self.selected_slot = None
        self.completed_booking = False
        self.paused = False
        self.api_blocked = False
        self._handoff_keys = set()
        self.turn_confirmed = False
        self.declined = False
        self.pending_slot_number = None
        self.changed = set()

    def handle(self, text):
        started = time.perf_counter()
        self._event = {"intent": None, "action": "handle", "outcome": "ok"}
        try:
            return self._handle(text)
        finally:
            self._event["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            logger.info(json.dumps(self._event, separators=(",", ":")))

    def _handle(self, text):
        self.turn_confirmed = False
        self.declined = False
        self.pending_slot_number = None
        self.changed = set()

        try:
            understood = self._understand(text)
        except UnderstandingError:
            self._mark("understand", "error")
            return "I couldn't interpret that message. Nothing was booked."

        if not isinstance(understood, dict):
            self._mark("understand", "invalid")
            return "I couldn't interpret that message. Nothing was booked."

        # Exact yes/no only counts after a real slot is already selected.
        # The model's confirmed flag still applies to every other reply.
        explicit = _explicit_confirmation(text) if self.selected_slot is not None else None
        held_slot = self.selected_slot
        self._apply(understood)
        if explicit is True:
            self.turn_confirmed = True
            self.declined = False
            self.selected_slot = held_slot
        elif explicit is False:
            self.turn_confirmed = False
            self.declined = True
            self.selected_slot = None
        turn_intent = understood.get("intent")
        if turn_intent and turn_intent != "unclear":
            self._event["intent"] = turn_intent
        elif self.intent:
            self._event["intent"] = self.intent

        if turn_intent in SPECIAL_INTENTS:
            self.paused = True
            self.selected_slot = None
            reason, summary, lead = SPECIAL_INTENTS[turn_intent]
            self._mark("handoff", reason)
            return "{} {}".format(lead, self._handoff(reason, summary))

        if self.changed or turn_intent in WORKFLOW_INTENTS:
            self.api_blocked = False
        elif self.api_blocked:
            self._mark("stop", "api_unavailable")
            return (
                "The scheduling system was unavailable, so I stopped. "
                "Nothing was booked. You can change the request, or ask to speak with a person."
            )

        if turn_intent in WORKFLOW_INTENTS:
            self.paused = False
            if turn_intent != self.intent:
                self._clear_identity()
                self.intent = turn_intent

        if self.paused:
            self._mark("handoff", "waiting")
            return "A staff member will follow up. Tell me if you want to start a new request."

        if self.intent == "provider_lookup":
            return self._provider_lookup()
        if self.intent == "book_appointment":
            return self._book()
        if self.intent == "appointment_lookup":
            return self._appointment_lookup()

        self._mark("clarify", "need_intent")
        return (
            "I can look up providers, book an appointment, or look up an existing appointment."
        )

    def _mark(self, action, outcome):
        event = getattr(self, "_event", None)
        if not isinstance(event, dict):
            return
        if not event.get("intent"):
            event["intent"] = self.intent
        event["action"] = action
        event["outcome"] = outcome

    def _apply(self, understood):
        for field in ("specialty", "location", "phone", "dob", "zip_code"):
            value = understood.get(field)
            if value is None or value == getattr(self, field):
                continue
            setattr(self, field, value)
            self.changed.add(field)
            if field in {"phone", "dob"}:
                self._clear_identity()
            elif field == "zip_code":
                self.patient = None
                self._clear_slots()
            elif field in {"specialty", "location"}:
                self._clear_slots()

        slot_number = understood.get("slot_number")
        if isinstance(slot_number, int):
            self.pending_slot_number = slot_number

        if understood.get("confirmed") is True:
            self.turn_confirmed = True
        elif understood.get("confirmed") is False:
            self.declined = True
            self.selected_slot = None

    def _provider_lookup(self):
        if not self.specialty and not self.location:
            self._mark("provider_lookup", "need_filters")
            return "Which specialty or location should I use? Specialties are primary care and dermatology. Locations are downtown, uptown, and lakeside."

        try:
            payload = lookup_providers(specialty=self.specialty, location=self.location)
        except SchedulingAPIError:
            return self._api_failure("look up providers")

        providers = payload.get("providers") if isinstance(payload, dict) else None
        if not isinstance(providers, list):
            return self._api_failure("look up providers")
        if not providers:
            self._mark("provider_lookup", "no_match")
            return "No providers matched that search."

        lines = ["Providers:"]
        for provider in providers:
            if not isinstance(provider, dict):
                continue
            name = provider.get("name") or "Unnamed provider"
            specialty = _words(provider.get("specialty"))
            locations = ", ".join(provider.get("locations") or [])
            modalities = ", ".join(_words(item) for item in (provider.get("modalities") or []))
            lines.append("- {} — {} — {} — {}".format(name, specialty, locations, modalities))
        if len(lines) == 1:
            self._mark("provider_lookup", "no_match")
            return "No providers matched that search."
        self._mark("provider_lookup", "ok")
        return "\n".join(lines)

    def _book(self):
        if self.completed_booking and not self.changed and self.pending_slot_number is None and not self.turn_confirmed:
            self._mark("book_appointment", "already_booked")
            return "That appointment is set. Tell me if you want a different time."
        if not self.specialty:
            self._mark("book_appointment", "need_specialty")
            return "Is this primary care or dermatology?"
        if not self.location:
            self._mark("book_appointment", "need_location")
            return "Which location: downtown, uptown, or lakeside?"

        waiting = self._ensure_patient()
        if waiting:
            return waiting

        if not self.slots_shown:
            return self._fetch_slots()

        if not self.slots:
            self._mark("book_appointment", "no_availability")
            return (
                "Nothing is available for {} at {}. ".format(_words(self.specialty), self.location)
                + "I can't offer a time the scheduling system didn't return."
            )

        if self.declined:
            self._mark("book_appointment", "declined")
            return "Okay, I won't book that.\n" + self._slot_prompt()

        if self.pending_slot_number is not None:
            problem = self._select_slot(self.pending_slot_number)
            if problem:
                return problem

        if self.turn_confirmed:
            if self.selected_slot is None:
                self._mark("book_appointment", "need_slot")
                return "Tell me the number of the time you want. I won't book until you pick one from the list."
            return self._book_selected()

        if self.selected_slot is not None:
            self._mark("book_appointment", "need_confirmation")
            return self._ask_confirmation()
        self._mark("book_appointment", "slots_offered")
        return self._slot_prompt()

    def _appointment_lookup(self):
        waiting = self._ensure_patient()
        if waiting:
            return waiting

        patient_id = self.patient.get("patientId")
        try:
            payload = lookup_appointments(patient_id)
        except SchedulingAPIError:
            return self._api_failure("look up appointments")

        appointments = payload.get("appointments") if isinstance(payload, dict) else None
        if not isinstance(appointments, list):
            return self._api_failure("look up appointments")
        lines = ["Appointments:"]
        for item in appointments:
            if not isinstance(item, dict):
                continue
            lines.append(
                "- {} — {} — {} — {}".format(
                    _when(item.get("startTime")),
                    _words(item.get("specialty")),
                    item.get("location") or "",
                    item.get("status") or "",
                )
            )
        if len(lines) == 1:
            self._mark("appointment_lookup", "no_appointments")
            return "I don't see any appointments for that patient."
        self._mark("appointment_lookup", "ok")
        return "\n".join(lines)

    def _ensure_patient(self):
        if not self.phone:
            self._mark(self.intent or "identify", "need_phone")
            return "What phone number should I use?"
        if not self.dob:
            self._mark(self.intent or "identify", "need_dob")
            return "What is the date of birth?"
        if self.patient is not None:
            return None
        if self.candidates is None:
            try:
                payload = search_patients(self.phone, self.dob)
            except SchedulingAPIError:
                return self._api_failure("search for a patient")
            matches = payload.get("matches") if isinstance(payload, dict) else None
            if not isinstance(matches, list):
                return self._api_failure("search for a patient")
            self.candidates = [
                match
                for match in matches
                if isinstance(match, dict) and isinstance(match.get("patientId"), str) and match.get("patientId")
            ]
        return self._resolve_candidates()

    def _resolve_candidates(self):
        matches = self.candidates or []
        chosen = matches
        if self.zip_code:
            chosen = [match for match in matches if match.get("zipCode") == self.zip_code]

        if len(chosen) == 1:
            self.patient = chosen[0]
            return None
        if not matches:
            note = self._handoff(
                "identity_unclear",
                "No patient matched the supplied phone number and date of birth.",
                once_key=("no_patient", self.phone, self.dob),
            )
            self._mark(self.intent or "identify", "no_patient")
            return (
                "I couldn't find a patient with that phone number and date of birth. "
                "I won't guess. You can give a corrected phone number and date of birth. "
                + note
            )
        if not chosen:
            self._mark(self.intent or "identify", "zip_mismatch")
            return (
                "That ZIP code doesn't match the patients I found. "
                "I won't pick one. You can give a different ZIP code, or ask to speak with a person."
            )
        if not self.zip_code:
            self._mark(self.intent or "identify", "need_zip")
            return (
                "I found more than one patient with that phone number and date of birth. "
                "What is your ZIP code?"
            )
        note = self._handoff(
            "identity_unclear",
            "More than one patient still matched after the ZIP code.",
            once_key=("ambiguous", self.phone, self.dob, self.zip_code),
        )
        self._mark(self.intent or "identify", "ambiguous_patient")
        return "More than one patient still matches, so I can't continue. " + note

    def _fetch_slots(self):
        patient_id = self.patient.get("patientId") if self.patient else None
        if not patient_id:
            self._mark("book_appointment", "missing_patient")
            return "I don't have a patient from the scheduling system, so I can't check openings."
        try:
            payload = lookup_availability(
                patient_id,
                self.specialty,
                location=self.location,
            )
        except SchedulingAPIError:
            return self._api_failure("check availability")

        raw_slots = payload.get("slots") if isinstance(payload, dict) else None
        if not isinstance(raw_slots, list):
            return self._api_failure("check availability")

        self.slots = [
            slot
            for slot in raw_slots
            if isinstance(slot, dict)
            and isinstance(slot.get("slotId"), str)
            and slot.get("slotId")
            and slot.get("available") is not False
        ]
        self.slots_shown = True
        self.selected_slot = None
        if not self.slots:
            self._mark("book_appointment", "no_availability")
            return (
                "Nothing is available for {} at {}. ".format(_words(self.specialty), self.location)
                + "I can't offer a time the scheduling system didn't return. "
                + "You can try another location or specialty, or ask to speak with a person."
            )
        self._mark("book_appointment", "slots_offered")
        return self._slot_prompt()

    def _select_slot(self, number):
        if number < 1 or number > len(self.slots):
            self.selected_slot = None
            self._mark("book_appointment", "invalid_slot")
            return "Choose a number from 1 to {}.\n{}".format(len(self.slots), self._format_slots())
        self.selected_slot = self.slots[number - 1]
        return None

    def _book_selected(self):
        slot = self.selected_slot
        patient_id = self.patient.get("patientId") if self.patient else None
        slot_id = slot.get("slotId") if isinstance(slot, dict) else None
        if slot not in self.slots or not patient_id or not slot_id:
            self._mark("book_appointment", "not_booked")
            return "That opening isn't one the scheduling system returned. I did not book an appointment."

        try:
            payload = book_appointment(patient_id, slot_id, True)
        except SchedulingAPIError as exc:
            if exc.status_code == 409:
                return self._slot_taken(slot_id)
            return self._api_failure("book the appointment")

        appointment = payload.get("appointment") if isinstance(payload, dict) else None
        if not isinstance(appointment, dict) or not appointment.get("appointmentId"):
            self._mark("book_appointment", "not_booked")
            return "The scheduling system did not return a booked appointment. I am not treating it as booked."

        self.completed_booking = True
        self.selected_slot = None
        self.slots = []
        self.slots_shown = False
        self._mark("book_appointment", "booked")
        return "Your appointment is {} for {} at {} ({}).".format(
            appointment.get("status") or "created",
            _when(appointment.get("startTime")),
            appointment.get("location") or "the scheduled location",
            _words(appointment.get("specialty")),
        )

    def _slot_taken(self, slot_id):
        self.slots = [slot for slot in self.slots if slot.get("slotId") != slot_id]
        self.selected_slot = None
        self.slots_shown = True
        self._mark("book_appointment", "slot_taken")
        lead = "That time was just taken. I did not book an appointment."
        if not self.slots:
            return lead + " Nothing else is available from the list I had."
        return lead + "\n" + self._slot_prompt()

    def _ask_confirmation(self):
        return "Book {}? Say yes to book it, or pick a different number.".format(
            _slot_text(self.selected_slot)
        )

    def _slot_prompt(self):
        return self._format_slots() + "\nTell me the number of the time you want."

    def _format_slots(self):
        lines = ["Available times:"]
        for number, slot in enumerate(self.slots, start=1):
            lines.append("{}. {}".format(number, _slot_text(slot)))
        return "\n".join(lines)

    def _api_failure(self, action):
        note = self._handoff(
            "api_failure",
            "Scheduling API error during {}.".format(action),
        )
        self.api_blocked = True
        self._mark(action, "api_error")
        return (
            "The scheduling system returned an error while I tried to {}. "
            "Nothing was booked. {}"
        ).format(action, note)

    def _handoff(self, reason, summary, once_key=None):
        if once_key is not None and once_key in self._handoff_keys:
            return "A staff member already has this."
        patient_id = None
        if self.patient and isinstance(self.patient.get("patientId"), str):
            patient_id = self.patient["patientId"]
        try:
            payload = create_handoff(reason, summary, patient_id=patient_id)
        except SchedulingAPIError:
            return "I couldn't queue a handoff to staff."
        if once_key is not None:
            self._handoff_keys.add(once_key)
        if isinstance(payload, dict) and payload.get("status") == "queued":
            return "I queued this for a staff member."
        return "I couldn't confirm that a handoff was queued."

    def _clear_slots(self):
        self.slots = []
        self.slots_shown = False
        self.selected_slot = None
        self.completed_booking = False

    def _clear_identity(self):
        self.patient = None
        self.candidates = None
        self._clear_slots()


_CONFIRM_YES = frozenset({
    "yes",
    "yes please",
    "confirm",
    "confirm it",
    "book it",
    "yes book it",
    "yes, book it",
})
_CONFIRM_NO = frozenset({
    "no",
    "no thanks",
    "don't book it",
    "do not book it",
    "cancel that",
})


def _explicit_confirmation(text):
    normalized = " ".join((text or "").strip().lower().split())
    if normalized in _CONFIRM_YES:
        return True
    if normalized in _CONFIRM_NO:
        return False
    return None


def _words(value):
    return str(value or "").replace("_", " ")


def _when(value):
    if not isinstance(value, str) or "T" not in value:
        return value or "an unknown time"
    date, rest = value.split("T", 1)
    return "{} {}".format(date, rest[:5])


def _slot_text(slot):
    return "{} — {} — {}".format(
        _when(slot.get("startTime")),
        _words(slot.get("specialty")),
        slot.get("location") or "",
    )
