"""Natural language understanding for one scheduling message.

This module only interprets user text. It does not call the scheduling API,
choose a workflow, book, hand off, or keep conversation state.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 20

INTENTS = (
    "provider_lookup",
    "book_appointment",
    "appointment_lookup",
    "human_request",
    "medical_advice",
    "unsupported",
    "unclear",
)
SPECIALTIES = ("primary_care", "dermatology")
LOCATIONS = ("downtown", "uptown", "lakeside")

_ZIP = re.compile(r"^\d{5}$")
_PHONE_CHARS = re.compile(r"^[\d\s().\-]+$")

_SYSTEM = """You interpret one scheduling message and return JSON only.
Do not answer the user. Do not give medical advice. Do not invent providers, patients, slots, times, phone numbers, or dates.

Return this object:
{
  "intent": "provider_lookup" | "book_appointment" | "appointment_lookup" | "human_request" | "medical_advice" | "unsupported" | "unclear",
  "specialty": "primary_care" | "dermatology" | null,
  "location": "downtown" | "uptown" | "lakeside" | null,
  "phone": string | null,
  "dob": string | null,
  "zip_code": string | null,
  "slot_number": number | null,
  "confirmed": true | false | null
}

Intent:
- provider_lookup: who the providers are, or where they practice
- book_appointment: the user needs to see, wants to see, schedule with, book with, or get an appointment with a doctor or provider. Also "book", "book appointment", "schedule appointment", and "make an appointment", including obvious misspellings of appointment such as "appoinemnt" or "appoinmetnen"
- appointment_lookup: wants an existing appointment
- human_request: ONLY when the user explicitly asks to speak with a human, person, agent, representative, staff member, or similar
- medical_advice: asks about symptoms, whether to wait, or for a diagnosis
- unsupported: ONLY when the user clearly asks for something outside supported scheduling, such as cardiology, billing, cancellation, or rescheduling
- unclear: the message does not independently express one of the intents above, or it is ambiguous rather than clearly unsupported

A message containing only a phone number, date of birth, ZIP code, confirmation, slot selection, specialty, or location does not create a new standalone intent. Return intent="unclear" unless the message independently expresses one. Still extract the field they stated.

"primary care", "PCP", "family doctor", and "regular doctor" are primary_care. "skin" and "dermatologist" are dermatology.
Allowed specialties are only primary_care and dermatology. Allowed locations are only downtown, uptown, and lakeside.
Tolerate obvious spelling mistakes. Map a misspelled specialty or location to one of those values only when it is unambiguous.
"primcary care", "primay care", and "primarycare" are primary_care. "dermatolgy" and "dermatlogy" are dermatology.
"dodnwtown", "dowtown", and "downtwon" are downtown. "lakesdie" is lakeside.
Do NOT use intent="unsupported" for an obvious typo of a supported specialty or location.
If a token could reasonably be more than one allowed value, use null and intent="unclear". Do not guess a location or ZIP code the user did not clearly mean.
"lookup" by itself is unclear. It does not say whether the user wants a provider or an existing appointment.

Examples:
"My phone is 555 0101" -> {"intent":"unclear","specialty":null,"location":null,"phone":"555-0101","dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"My ZIP is 70115" -> {"intent":"unclear","specialty":null,"location":null,"phone":null,"dob":null,"zip_code":"70115","slot_number":null,"confirmed":null}
"I'll take the second one" -> {"intent":"unclear","specialty":null,"location":null,"phone":null,"dob":null,"zip_code":null,"slot_number":2,"confirmed":null}
"I need to see a regular doctor downtown" -> {"intent":"book_appointment","specialty":"primary_care","location":"downtown","phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"I want to speak to a person" -> {"intent":"human_request","specialty":null,"location":null,"phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"book" -> {"intent":"book_appointment","specialty":null,"location":null,"phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"book an appoinemnt in dodnwtown" -> {"intent":"book_appointment","specialty":null,"location":"downtown","phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"schedule a dermatolgy appointment at lakeside" -> {"intent":"book_appointment","specialty":"dermatology","location":"lakeside","phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"lookup" -> {"intent":"unclear","specialty":null,"location":null,"phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"primcary care" -> {"intent":"unclear","specialty":"primary_care","location":null,"phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"primay care" -> {"intent":"unclear","specialty":"primary_care","location":null,"phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"primarycare" -> {"intent":"unclear","specialty":"primary_care","location":null,"phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"dermatolgy" -> {"intent":"unclear","specialty":"dermatology","location":null,"phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"dermatlogy" -> {"intent":"unclear","specialty":"dermatology","location":null,"phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"dowtown" -> {"intent":"unclear","specialty":null,"location":"downtown","phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"downtwon" -> {"intent":"unclear","specialty":null,"location":"downtown","phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"lakesdie" -> {"intent":"unclear","specialty":null,"location":"lakeside","phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"book an appointment in dowtown" -> {"intent":"book_appointment","specialty":null,"location":"downtown","phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"cardiology" -> {"intent":"unsupported","specialty":null,"location":null,"phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
"cancel my appointment" -> {"intent":"unsupported","specialty":null,"location":null,"phone":null,"dob":null,"zip_code":null,"slot_number":null,"confirmed":null}
phone is only a number the user stated. Keep every digit they said. Do not add or drop digits.
dob is the date-of-birth text the user stated. Copy that text. Do not reformat it or fill in a missing year, month, or day.
zip_code is the 5-digit ZIP the user stated. It distinguishes patients who share a name, phone, and date of birth.
slot_number is the option the user picked, as a number: "the first one" is 1, "slot 2" is 2. It is not an internal slot id. Use null if they did not pick one.
confirmed is true only when this message explicitly agrees to book, false when it explicitly declines, otherwise null.
"""


class UnderstandingError(Exception):
    """The model call failed or returned something this module cannot use."""


def understand(text: str, model: Optional[str] = None) -> dict:
    """Return intent and fields stated in this message."""
    message = (text or "").strip()
    if not message:
        return _blank("unclear")

    raw = _complete(message, model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL)
    return _normalize(raw)


def _complete(message: str, model: str) -> dict:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise UnderstandingError("OPENAI_API_KEY is not set.")

    client = OpenAI(timeout=DEFAULT_TIMEOUT_SECONDS)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": message},
            ],
        )
    except (APIError, APIConnectionError, APITimeoutError) as exc:
        raise UnderstandingError("Language model request failed.") from exc

    content = response.choices[0].message.content if response.choices else None
    try:
        parsed = json.loads(content or "")
    except (TypeError, json.JSONDecodeError) as exc:
        raise UnderstandingError("Language model returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise UnderstandingError("Language model returned invalid JSON.")
    return parsed


def _normalize(raw: dict) -> dict:
    intent = raw.get("intent")
    if intent not in INTENTS:
        intent = "unclear"

    specialty = raw.get("specialty")
    if specialty not in SPECIALTIES:
        specialty = None

    location = raw.get("location")
    if location not in LOCATIONS:
        location = None

    phone = _phone(raw.get("phone"))

    dob = _normalize_dob(raw.get("dob"))

    zip_code = raw.get("zip_code")
    if isinstance(zip_code, int):
        zip_code = str(zip_code)
    zip_code = zip_code.strip() if isinstance(zip_code, str) else ""
    zip_code = zip_code if _ZIP.match(zip_code) else None

    slot_number = _slot_number(raw.get("slot_number"))

    confirmed = raw.get("confirmed")
    if not isinstance(confirmed, bool):
        confirmed = None

    return {
        "intent": intent,
        "specialty": specialty,
        "location": location,
        "phone": phone,
        "dob": dob,
        "zip_code": zip_code,
        "slot_number": slot_number,
        "confirmed": confirmed,
    }


def _blank(intent: str) -> dict:
    return {
        "intent": intent,
        "specialty": None,
        "location": None,
        "phone": None,
        "dob": None,
        "zip_code": None,
        "slot_number": None,
        "confirmed": None,
    }


def _normalize_dob(value):
    """Return YYYY-MM-DD, or None when the text is incomplete, invalid, or ambiguous."""
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text:
        return None

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return _one_date(text, ("%Y-%m-%d",))
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", text):
        return _one_date(text, ("%m/%d/%Y",))
    if re.fullmatch(r"\d{1,2}-\d{1,2}-\d{4}", text):
        return _one_date(text, ("%m-%d-%Y",))
    return _one_date(text, ("%B %d %Y", "%B %d, %Y", "%b %d %Y", "%b %d, %Y"))


def _one_date(text, formats):
    found = set()
    for fmt in formats:
        try:
            found.add(datetime.strptime(text, fmt).date())
        except ValueError:
            continue
    if len(found) != 1:
        return None
    return found.pop().isoformat()


def _phone(value):
    """Format a stated 7-digit number as 555-0101. Leave anything else unchanged."""
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    digits = re.sub(r"\D", "", stripped)
    if _PHONE_CHARS.match(stripped) and len(digits) == 7:
        return "{}-{}".format(digits[:3], digits[3:])
    return stripped


def _slot_number(value):
    """Keep a user-stated option number. Drop internal slot ids."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value.isdigit():
            return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 1:
        return None
    return number
