# Appointment scheduling assistant

A multi-turn CLI that schedules appointments by calling the supplied mock scheduling API. The model only interprets a user message. Deterministic code decides what is missing, which API call to make, and whether a booking is allowed.

```text
app.py              CLI. One Orchestrator stays alive for the session.
llm.py              Natural language understanding only.
orchestrator.py     Workflow and safety decisions.
tools.py            Scheduling API client.
mock scheduling API Source of truth for providers, patients, slots, and bookings.
```

## Setup

Python 3.8 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="..."
```

`python-dotenv` also loads a local `.env` if one is present. Do not commit the key. The mock API defaults to `http://localhost:4010`. Override it with `SCHEDULING_API_BASE_URL`. The model defaults to `gpt-4o-mini`. Override it with `OPENAI_MODEL`.

## Run the mock API

In one terminal:

```bash
python3 mock-api/server.py
```

The API listens on `http://localhost:4010`. State is in memory. Restart the process to reset bookings.

## Run the CLI

In another terminal:

```bash
python3 app.py
```

Type `quit` to exit. An optional first argument is handled as the first turn, then the same session continues.

The CLI needs the mock API and `OPENAI_API_KEY`. Each line is one turn. Workflow logs are JSON lines on stderr. They record intent, action, API operation, outcome, and latency. They do not include phone, date of birth, ZIP, or patient records.

## Run evals

With the mock API already running:

```bash
python3 evals.py
```

Evals do not call the language model. They script understanding and assert against the real mock API. A passing run prints:

```text
PASS provider_lookup
PASS happy_path_booking
PASS no_patient
PASS no_availability
PASS slot_conflict
5/5 passed
```

The happy-path check books a real downtown primary care slot. Restart the mock API to clear it. If every non-conflict downtown slot is already taken, that check fails until the API is restarted.

## Supported flows

- Provider lookup for primary care or dermatology, optionally filtered by downtown, uptown, or lakeside.
- Book an appointment: collect specialty, location, phone, and date of birth, search the patient, show API availability, then book only after an explicit yes.
- Look up an existing patient's appointments after the same identity check.
- No patient match: say so, do not continue to availability or booking, and queue a handoff.
- More than one patient match: ask for a ZIP code and do not pick a patient until one match remains.
- No availability: say nothing is open and do not invent a time.
- Slot conflict: `slot_conflict_001` can be selected from availability, then booking returns HTTP 409. The assistant says the time was taken and does not report a booking.
- Scheduling API failure: report the error, do not claim success, and attempt a handoff.
- A request to speak to a person, a request for medical advice, or an unsupported request is handed off. The assistant does not give medical advice.

Cancel and reschedule are not implemented.

## Safety and design decisions

- The model never invents patient IDs or slot IDs. It returns intent and fields the user stated.
- A patient ID is used only after `GET /patients/search` returns that patient.
- A slot ID is used only if it is one of the slots from `GET /availability`. A spoken slot number is mapped onto that list.
- Booking calls `POST /appointments` with `confirmed: true` only after the user has seen the list, selected one of those slots, and explicitly confirmed. "yes" counts only once a slot is already selected.
- Zero matches and unresolved duplicate matches are not guessed.
- A failed or empty API response is not turned into a successful booking, a made-up provider, or a made-up time.
- Conversation state lives in memory on one `Orchestrator` for the prototype. It is gone when the process exits.

## Limitations

- One local CLI session. No login, no shared store, and no recovery after restart.
- Evals cover the workflow and the mock API. They do not grade model wording.
- Booking is a single POST. A retry could create a second appointment.
- The client does not retry failed API calls.
- Supported specialties are primary care and dermatology. Supported locations are downtown, uptown, and lakeside.

## Production next steps

- Authenticated sessions.
- Persistent state.
- Idempotency for booking writes.
- Retries with backoff.
- Observability and P95 latency monitoring.
- CI evaluation gates.
- Canary rollout and rollback.
