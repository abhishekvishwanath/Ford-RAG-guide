"""Immediate-hazard responses. These are stop-work safety messages, not repair instructions."""

from __future__ import annotations

HAZARD_MESSAGES = {
    "fire_or_smoke": (
        "Stop work now. If there is fire or smoke from the engine bay, wiring or battery, move everyone away from "
        "the vehicle and call emergency services. Do not open the hood on an active fire. Only use a suitable "
        "extinguisher if it is safe and you are trained to do so."
    ),
    "fuel_leak": (
        "Stop work now. Fuel leaks are a fire hazard: switch the ignition off, remove ignition sources, keep the "
        "area ventilated, do not create sparks (including connecting or disconnecting the battery), and contain "
        "the spill following your workshop's spill procedure."
    ),
    "brake_failure": (
        "Do not drive the vehicle. A loss of braking is an immediate safety risk: keep the vehicle parked and "
        "secured, and only move it by recovery or tow until the brake system has been inspected and repaired."
    ),
    "unstable_vehicle_support": (
        "Stop work and keep everyone clear of the vehicle. Do not reach under a vehicle that is unstable on a jack, "
        "stands or a lift. Re-establish safe support using the workshop's lifting procedure before continuing."
    ),
    "electrical_hazard": (
        "Stop work and keep clear. Do not touch arcing components or a swollen, hissing or smoking battery. "
        "Isolate power only if it can be done safely, ventilate the area, and follow your workshop's electrical "
        "safety procedure."
    ),
    "injury": ("Stop work and attend to the injured person first. Call emergency services if the injury is serious."),
}


def safety_message(hazard: str) -> str:
    body = HAZARD_MESSAGES.get(hazard, "Stop work and make the area safe before continuing.")
    return (
        f'⚠️ {body}\n\nWhen the situation is safe and under control, tell me (for example, "it\'s safe now") and '
        "I'll continue with the workshop manual information. I can't confirm the vehicle is safe to drive."
    )
