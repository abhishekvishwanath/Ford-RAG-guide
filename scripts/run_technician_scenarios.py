"""Ten hard, technician-style conversations against a running API (live Groq calls).

  python scripts/run_technician_scenarios.py out.json [CASE_ID ...]

Each case has scripted opening turns plus how the technician answers if the assistant asks for a
detail (by slot), so the conversation adapts to the assistant's questions. Waits 45 s after every
answer because of the Groq on-demand rate limits. See docs/evaluation_report.md §8 for findings."""
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = os.environ.get("ASSISTANT_API", "http://localhost:8000/api/chat")
OUT = sys.argv[1] if len(sys.argv) > 1 else "technician_scenarios.json"
ONLY = set(sys.argv[2:])

# turns: scripted technician messages. slots: how the tech answers if the bot asks.
CASES = [
    {"id": "T01_gt_plugs", "turns": ["need to do the plugs on a GT, they keep fouling. what do they torque to?",
                                     "and what gap should they have?"],
     "slots": {"model_year": "2014", "engine": "yes it's the 5.0", "transmission": "manual"}},
    {"id": "T02_brake_pads_variant", "turns": ["front pads are down to the backing plate on a 2014 GT, how do I swap them",
                                               "what do the caliper bolts torque to?"],
     "slots": {"engine": "5.0L", "transmission": "automatic", "body_style": "coupe", "component": "it has the 4 piston brembo calipers"}},
    {"id": "T03_clutch_5_0", "turns": ["2014 mustang 5.0 six speed, clutch is slipping. walk me through replacing the clutch disc and pressure plate"],
     "slots": {"body_style": "coupe"}},
    {"id": "T04_gt500_belt", "turns": ["Shelby GT500, 2014. accessory belt is squealing, how do i replace it?",
                                       "is there a routing diagram?"],
     "slots": {"engine": "yes 5.8", "transmission": "manual"}},
    {"id": "T05_v6_atf_level", "turns": ["2014 V6 auto. how do I check the transmission fluid level? there's no dipstick"],
     "slots": {"engine": "3.7L", "body_style": "convertible"}},
    {"id": "T06_fuel_smell_safety", "turns": ["strong fuel smell and a puddle under the back of the car, I want to pull the fuel pump module on a 2014 5.0",
                                              "car is outside, battery is disconnected and the leak is contained. continue"],
     "slots": {"transmission": "manual", "body_style": "coupe"}},
    {"id": "T07_out_of_scope_then_fix", "turns": ["rear axle seal is leaking on my 2011 mustang gt",
                                                  "sorry, it's actually a 2014 GT 5.0. same job, rear wheel bearing and axle seal"],
     "slots": {"transmission": "manual", "body_style": "coupe"}},
    {"id": "T08_timing_followups", "turns": ["2014 5.0L, replacing the timing chains. what's the procedure",
                                             "what special tools do I need for this?",
                                             "torque on the camshaft sprocket bolts?"],
     "slots": {"transmission": "manual", "body_style": "coupe"}},
    {"id": "T09_alignment_specs", "turns": ["doing a front camber adjustment on a 2014 mustang, how is it adjusted and what's the spec?"],
     "slots": {"engine": "5.0L", "transmission": "manual", "body_style": "coupe"}},
    {"id": "T10_conflict_v6_gt", "turns": ["2014 GT with the 3.7 v6, need to replace the starter motor"],
     "slots": {"engine": "it's actually a V6, the 3.7L", "transmission": "automatic", "body_style": "coupe", "model_year": "2014"}},
]


def post(body):
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r), time.time() - t
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode()[:500]}, time.time() - t


def run(case):
    sid, log, queue = None, [], list(case["turns"])
    guard = 0
    while queue and guard < 10:
        guard += 1
        msg = queue.pop(0)
        resp, dt = post({"session_id": sid, "message": msg} if sid else {"message": msg})
        if "error" in resp:
            log.append({"user": msg, "error": resp})
            break
        sid = resp["session_id"]
        entry = {
            "user": msg, "secs": round(dt, 1), "mode": resp["interaction_mode"],
            "assistant": resp["assistant_message"], "readiness": resp["readiness"],
            "pending": resp.get("pending_question"),
            "vehicle": {k: (v or {}).get("value") for k, v in resp["vehicle_context"].items()},
            "repair": {k: (v or {}).get("value") for k, v in resp["repair_context"].items()},
            "sources": [(s.get("procedure_title"), s.get("page_start"), s.get("applicability_note")) for s in resp["retrieved_sources"]],
            "citations": resp["citations"], "images": [(i.get("caption"), i.get("page")) for i in resp["images"]],
            "validation": resp.get("validation"),
        }
        log.append(entry)
        print(case["id"], "|", msg[:60], "->", resp["interaction_mode"], flush=True)
        pq = resp.get("pending_question")
        if pq and resp["interaction_mode"] in ("asking_context", "awaiting_confirmation"):
            ans = None
            for s in pq.get("slots", []):
                if s in case["slots"]:
                    ans = case["slots"][s]
                    break
            if ans is None and pq.get("kind") == "confirmation":
                ans = "yes"
            if ans is None:
                ans = "not sure"
            queue.insert(0, ans)
        if resp["retrieved_sources"]:
            time.sleep(45)  # Groq 8k TPM
    return {"id": case["id"], "session_id": sid, "turns": log}


if __name__ == "__main__":
    results = [run(c) for c in CASES if not ONLY or c["id"] in ONLY]
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=1)
    print("done")
