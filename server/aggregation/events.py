"""Merge runs of flagged samples/windows into timed events (shared schema, flow.txt §A.2 / §B.6)."""


def events_from_flags(slots, flags, vtype, source, peaks=None, details=None, gap_s=0.05):
    """
    slots   : [(start_s, end_s)] time owned by each sample/window (non-overlapping, sorted)
    flags   : [bool] final (persistence-filtered) flag per slot
    peaks   : optional [float] per slot (score/confidence); event peak = max over the run
    details : optional callable(list_of_indices) -> dict
    Consecutive flagged slots (gap <= gap_s) become one event.
    """
    events, run = [], []

    def close():
        if not run:
            return
        a, b = slots[run[0]][0], slots[run[-1]][1]
        ev = {"type": vtype, "source": source, "start_s": round(a, 2), "end_s": round(b, 2),
              "duration_s": round(b - a, 2), "n": len(run), "first_index": run[0]}
        if peaks is not None:
            vals = [peaks[i] for i in run if peaks[i] is not None]
            ev["peak"] = round(max(vals), 3) if vals else None
        if details is not None:
            ev["details"] = details(list(run))
        events.append(ev)

    # Unflagged slots do not end an event by themselves: the next flagged slot joins the current event when
    # the time gap to it is <= gap_s (VAD3 fix: the previous loop closed the event at every unflagged slot,
    # so gap_s never merged anything). With the default 0.05 s, one unflagged slot still splits an event.
    for i, f in enumerate(flags):
        if not f:
            continue
        if run and slots[i][0] - slots[run[-1]][1] <= gap_s:
            run.append(i)
        else:
            close()
            run = [i]
    close()
    return events


def video_slots(samples, rate_hz):
    """Each video sample owns [t, t + 1/rate) (clipped to the next sample's time)."""
    out = []
    for i, s in enumerate(samples):
        end = samples[i + 1]["t"] if i + 1 < len(samples) else s["t"] + 1.0 / rate_hz
        out.append((s["t"], min(end, s["t"] + 1.0 / rate_hz)))
    return out
