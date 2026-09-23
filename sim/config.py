"""Static configuration: machine specs, sensor channels, alarm thresholds, economics.

Numbers are representative of a 336-class hydraulic excavator and a 730-class
articulated truck. They are plausible, not manufacturer data.
"""

SIM_DT_S = 1.0

SOIL_DENSITY_T_PER_M3 = 1.8
FUEL_PRICE_PER_L = 1.05  # USD
CO2_KG_PER_L_DIESEL = 2.68

EXCAVATOR_SPEC = {
    "model_class": "336-class hydraulic excavator",
    "fuel_tank_l": 600.0,
    "def_tank_l": 60.0,
    "bucket_m3": 1.9,
    "bucket_linkage_t": 3.6,  # bucket + linkage + effective boom/stick mass referred to the tip
    "low_idle_rpm": 950.0,
    "work_rpm": 1800.0,
    "fuel_idle_lph": 4.2,
    "fuel_span_lph": 33.8,  # added at 100 % load
    "def_ratio": 0.035,  # DEF litres per diesel litre
    "tail_swing_m": 3.6,
    "max_reach_m": 10.5,
    "rated_moment_tm": 80.0,  # tipping-limited load moment on level ground
    "service_interval_h": 500.0,
    "travel_speed_mps": 1.0,
}

TRUCK_SPEC = {
    "model_class": "730-class articulated truck",
    "fuel_tank_l": 400.0,
    "payload_t": 28.0,
    "target_payload_t": 25.0,
    "fuel_idle_lph": 4.0,
    "fuel_span_lph": 40.0,
    "loaded_speed_kmh": 22.0,
    "empty_speed_kmh": 30.0,
}

# name -> (unit, description, J1939-style SPN or None)
CHANNELS = {
    "engine_rpm": ("rpm", "Engine speed", 190),
    "engine_load_pct": ("%", "Engine percent load at current speed", 92),
    "fuel_rate_lph": ("L/h", "Engine fuel rate", 183),
    "fuel_level_pct": ("%", "Fuel tank level", 96),
    "def_level_pct": ("%", "DEF tank level", 1761),
    "coolant_temp_c": ("°C", "Engine coolant temperature", 110),
    "oil_pressure_kpa": ("kPa", "Engine oil pressure", 100),
    "oil_temp_c": ("°C", "Engine oil temperature", 175),
    "fuel_temp_c": ("°C", "Fuel temperature", 174),
    "egt_c": ("°C", "Exhaust gas temperature", 173),
    "air_filter_dp_kpa": ("kPa", "Air filter differential pressure", 107),
    "water_in_fuel_pct": ("%", "Water separator bowl level", 97),
    "battery_v": ("V", "Battery / system voltage (24 V system)", 168),
    "hyd_oil_temp_c": ("°C", "Hydraulic oil temperature", None),
    "hyd_pressure_mpa": ("MPa", "Main pump pressure", None),
    "trans_pressure_kpa": ("kPa", "Transmission oil pressure", 127),
    "pitch_deg": ("°", "Chassis pitch (IMU)", None),
    "roll_deg": ("°", "Chassis roll (IMU)", None),
    "swing_angle_deg": ("°", "Upper structure swing angle", None),
    "swing_speed_dps": ("°/s", "Swing speed", None),
    "reach_m": ("m", "Bucket reach from swing centre", None),
    "boom_tip_height_m": ("m", "Bucket tip height above grade", None),
    "max_height_m": ("m", "Highest point of front linkage", None),
    "payload_t": ("t", "Bucket payload (payload system)", None),
    "load_moment_pct": ("%", "Load moment vs tipping limit (slope compensated)", None),
    "travel_speed_kmh": ("km/h", "Ground speed", 84),
    "vibration_g": ("g", "Cab vibration RMS", None),
    "impact_g": ("g", "Peak shock this second", None),
    "control_activity": ("0-1", "Joystick activity", None),
    "control_jerk": ("0-1", "Joystick jerk (smoothness, lower is smoother)", None),
    "cab_temp_c": ("°C", "Cab air temperature", None),
    "ambient_temp_c": ("°C", "Ambient air temperature", 171),
    "humidity_pct": ("%", "Relative humidity", None),
    "seat_occupied": ("bool", "Operator seat switch", None),
    "seatbelt_fastened": ("bool", "Seatbelt buckle switch", None),
    "end_stop_hit": ("bool", "Hydraulic cylinder reached end of stroke under pressure", None),
    "engine_hours": ("h", "Engine hour meter", 247),
}

# Fixed alarm thresholds (L1). Oil pressure is speed dependent and handled in code.
THRESHOLDS = {
    "coolant_temp_c": {"warn_high": 100.0, "crit_high": 105.0},
    "hyd_oil_temp_c": {"warn_high": 90.0, "crit_high": 98.0},
    "egt_c": {"warn_high": 600.0, "crit_high": 650.0},
    "fuel_level_pct": {"warn_low": 15.0, "crit_low": 8.0},
    "def_level_pct": {"warn_low": 10.0, "crit_low": 5.0},
    "water_in_fuel_pct": {"warn_high": 50.0, "crit_high": 80.0},
    "air_filter_dp_kpa": {"warn_high": 5.0, "crit_high": 6.2},
    "battery_v": {"warn_low": 26.0, "crit_low": 25.0, "running_only": True},
    "trans_pressure_kpa": {"warn_low": 1200.0, "crit_low": 800.0, "running_only": True},
}

OIL_PRESSURE_MIN_KPA = [  # (rpm_at_least, warn_below, crit_below)
    (1400.0, 230.0, 180.0),
    (0.0, 140.0, 100.0),
]

SAFETY = {
    "seatbelt_grace_s": 15,
    "unattended_after_s": 90,
    "proximity_buffer_m": 1.0,
    "proximity_warn_extra_m": 6.0,
    "load_moment_warn_pct": 85.0,
    "load_moment_crit_pct": 100.0,
    "tilt_warn_deg": 15.0,
    "powerline_warn_margin_m": 1.0,
    "lightning_warn_km": 16.0,
    "lightning_crit_km": 10.0,
    "wind_gust_warn_kmh": 50.0,
}
