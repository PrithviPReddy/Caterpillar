import time, collections
from sim.scenarios import get_scenario
from sim.world import World

w = World(get_scenario("demo_day"))
t0 = time.time(); n = 0
states = collections.Counter()
while not w.done:
    frames = w.step()
    n += 1
    e = w.machines["EXC001"]
    states[(e.activity)] += 1
    if w.t.second == 0 and w.t.minute % 20 == 0:
        te = e.telemetry; t2 = w.machines["EXC002"].telemetry
        trk = " ".join(f"{m.id}:{m.state[:5]}" for m in w.machines.values() if m.kind == "truck")
        print(f"{w.t:%H:%M} E1 {e.activity:10s} {te['state']:12s} task={te['task_id']} {te['task_done_m3']}/{te['task_volume_m3']} "
              f"fuel={te['fuel_level_pct']:5.1f} cool={te['coolant_temp_c']:5.1f} hyd={te['hyd_oil_temp_c']:4.1f} "
              f"lm={te['load_moment_pct']:5.1f} maxh={te['max_height_m']} | E2 {t2['state']:8s} {t2['task_done_m3']} wif={t2['water_in_fuel_pct']} afdp={t2['air_filter_dp_kpa']} | {trk}")
print(f"{n} ticks in {time.time()-t0:.1f}s -> {(time.time()-t0)/n/len(w.machines)*1e6:.0f} us/machine-tick")
print(states)
for m in w.machines.values():
    if m.kind == "excavator":
        print(m.id, [(tk['id'], round(tk['done_m3']), tk['started_at'] and tk['started_at'].strftime('%H:%M'), tk['finished_at'] and tk['finished_at'].strftime('%H:%M')) for tk in m.tasks])
