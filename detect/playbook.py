"""Baseline actions per event type and the training modules behaviours map to.

These are the deterministic fallback shown when no agent reply is available. The
LangGraph agent is expected to adapt them to the situation.
"""

TRAINING_MODULES = {
    "TM-01": "Pre-start checks and engine warm-up",
    "TM-02": "Seatbelt use and safe cab entry/exit",
    "TM-03": "Smooth hydraulic control: feathering, avoiding end-stops",
    "TM-04": "Idle reduction and auto-idle",
    "TM-05": "Engine cool-down and turbocharger care",
    "TM-06": "Working on slopes and reading the lift chart",
    "TM-07": "Working near overhead power lines",
    "TM-08": "Swing radius and ground crew awareness",
    "TM-09": "Fatigue and heat stress management",
    "TM-10": "Daily fluid checks and water separator draining",
    "TM-11": "Haul road speed discipline around pedestrians",
}

PLAYBOOK = {
    "COOLANT_TEMP_HIGH": (["Reduce load and let the engine idle at mid speed to cool",
                           "Do not open the radiator cap while hot",
                           "Check radiator/cooler core for debris and coolant level once cooled"], None),
    "COOLANT_TEMP_RISING": (["Ease off heavy digging to slow the temperature rise",
                             "Plan a stop to inspect the cooling package before the alarm"], None),
    "COOLING_DEGRADATION_DETECTED": (["Schedule a cooling-system inspection at the next break",
                                      "Blow out radiator and hydraulic cooler cores",
                                      "Check fan belt tension and coolant level"], None),
    "OIL_PRESSURE_LOW": (["Stop the engine as soon as it is safe", "Check oil level; do not restart if low"], None),
    "OIL_PRESSURE_DEVIATION": (["Check engine oil level and look for leaks",
                                "Book oil pump / pressure sender check"], None),
    "FUEL_EFFICIENCY_DEGRADATION": (["Check air filter and intake for restriction",
                                     "Book injector / turbo inspection"], None),
    "HYD_OIL_TEMP_HIGH": (["Reduce cycle intensity; avoid holding functions over relief",
                           "Check hydraulic cooler for debris"], "TM-03"),
    "HYD_TEMP_DEVIATION": (["Inspect hydraulic cooler and oil level"], None),
    "HYD_OIL_TEMP_RISING": (["Avoid holding functions over relief", "Check hydraulic cooler"], "TM-03"),
    "EGT_HIGH": (["Reduce load", "Check for intake restriction"], None),
    "FUEL_LOW": (["Arrange refuelling", "Avoid running the tank dry (air in fuel system)"], None),
    "DEF_LOW": (["Top up DEF; engine will derate when empty"], None),
    "WATER_IN_FUEL": (["Drain the water separator now", "Report the fuel batch/bowser for checking"], "TM-10"),
    "WATER_IN_FUEL_RISING": (["Drain the water separator at the next stop",
                              "Suspect contaminated fuel delivery"], "TM-10"),
    "AIR_FILTER_RESTRICTED": (["Replace the primary air filter element"], None),
    "BATTERY_VOLTAGE_LOW": (["Check alternator output and belt"], None),
    "TRANSMISSION_PRESSURE_LOW": (["Stop the truck safely off the haul road", "Call the mechanic; do not drive"],
                                  None),
    "SENSOR_FLATLINE": (["Flag the sensor for replacement", "Treat related alarms as unreliable until fixed"],
                        None),
    "FUEL_LOSS_ENGINE_OFF": (["Check the machine and surroundings for siphoning or leaks",
                              "Secure the fuel cap; review site CCTV", "Report to site security"], None),
    "ENGINE_START_OUTSIDE_SHIFT": (["Verify who is operating the machine"], None),
    "SEATBELT_UNFASTENED": (["Stop, fasten the seatbelt, then continue"], "TM-02"),
    "UNATTENDED_MACHINE": (["Shut the engine down when leaving the cab", "Lower attachments to the ground"],
                           "TM-04"),
    "PROXIMITY_DANGER_ZONE": (["STOP all motion now", "Wait until the person is clear of the swing radius",
                               "Re-establish eye contact / radio with ground crew"], "TM-08"),
    "PROXIMITY_WARNING": (["Slow down and confirm the ground worker has seen you"], "TM-08"),
    "STABILITY_LIMIT": (["Reduce reach and bring the load closer", "Avoid swinging loaded downhill",
                         "Position the machine square to the slope"], "TM-06"),
    "SLOPE_TILT": (["Reposition to a flatter bench"], "TM-06"),
    "POWERLINE_CLEARANCE": (["Lower the boom immediately", "Keep all parts below the height limit",
                             "Use a spotter under overhead lines"], "TM-07"),
    "NO_DIG_ZONE": (["Stop digging: buried utility", "Contact site engineer for a permit to dig"], None),
    "OVERSPEED": (["Slow to the posted limit"], "TM-11"),
    "LIGHTNING_NEARBY": (["Lower booms and stop lifting", "Ground crew to shelter in vehicles/buildings",
                          "Operators stay inside enclosed cabs"], None),
    "HIGH_WIND": (["Suspend lifting operations"], None),
    "WARMUP_VIOLATION": (["Idle 5 min after a cold start before heavy digging"], "TM-01"),
    "HOT_SHUTDOWN": (["Idle 3-5 min before shutting down after heavy work"], "TM-05"),
    "HARSH_OPERATION": (["Feather the controls near the end of stroke", "Avoid bucket slamming"], "TM-03"),
    "EXCESSIVE_IDLE": (["Shut down if waiting more than 5 min", "Use auto-idle"], "TM-04"),
    "TRUCK_WAIT_IDLE": (["Dispatch: rebalance trucks or add a truck",
                         "Operator: use waiting time for bench prep / clean-up"], None),
    "FATIGUE_RISK": (["Take a 10 min break out of the cab", "Hydrate"], "TM-09"),
    "HEAT_STRESS": (["Hydrate every 15-20 min", "Check cab A/C", "Rotate or take a cool-down break"], "TM-09"),
    "FUEL_SHORTFALL": (["Refuel at the suggested break", "Book the fuel bowser now"], None),
    "DEF_SHORTFALL": (["Top up DEF at the next break"], None),
    "SERVICE_DUE": (["Book the service slot with maintenance"], None),
    "SERVICE_OVERDUE": (["Stop for scheduled service as soon as practical"], None),
    "MAINTENANCE_FORECAST": (["Order the part and schedule replacement before the limit"], None),
    "TASK_DELAY": (["Review truck allocation / sequence of tasks", "Update the client/foreman on the new ETA"],
                   None),
    "WEATHER_WINDOW": (["Finish or make safe weather-sensitive work before the rain",
                        "Swap to a task that tolerates rain"], None),
    "SHIFT_BRIEFING": (["Review today's plan and the flagged items before starting"], None),
    "SHIFT_SUMMARY": (["Review the scorecard with the operator"], None),
}


def actions_for(event_type):
    return list(PLAYBOOK.get(event_type, ([], None))[0])


def training_for(event_type):
    tm = PLAYBOOK.get(event_type, ([], None))[1]
    return {"id": tm, "title": TRAINING_MODULES[tm]} if tm else None
