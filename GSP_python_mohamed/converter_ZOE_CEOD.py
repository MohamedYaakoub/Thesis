attributes = [
    'Ps0 Ambient Static Pressure',
    'Ts0 Ambient Static Temperature',
    'Selected Total Temperature at Station 12 (DEG_C)',
    'Selected PT2 Pressure (PSIA)',
    'Selected HP Comp Inlet Total Temperature (DEG_C)',
    'Selected Compressor Delay Total Temperature (DEG_C)',
    'Selected Compressor Discharge Static Pressure (PSIA)',
    'Selected Exhaust Gas Temperature (DEG_C)',
    'Selected Fan Speed (%)',
    'Selected Core Speed (%)',
    'Selected Mass Fuel Flow (PPH)',
    'HP Turbine Active Clearance Control Valve Position',
    'LP Turbine Active Clearance Control Valve Position',
    'Transient Bleed Valve (TBV) Position',
    'Selected Variable Bleed Valve (VBV) Position',
    'Selected Variable Stator Vane (VSV) Position (%)',
    'Total Engine Horsepower Extraction (HP)',
    'Cowl Anti Ice (CAI) Bleed Configuration',
    'Booster Anti Ice (BAI) pressure',
    'Calculated HPT Clearance',
    'Synthesized core Mass Flow',
    'Core Speed Rate of Change (%N2/SEC)',
    'Corrected Fan Speed to Station 12 (%)',
    'Corrected Core Speed',
    'EGT probe 1',
    'EGT probe 2',
    'EGT probe 3',
    'EGT probe 4',
    'EGT probe 5',
    'EGT probe 6',
    'EGT probe 7',
    'EGT probe 8',
    'Peak EGT Value',
    'Peak EGT Value Takeoff',
    'UVL_FLIGHTPHS',
    'Offset',
    'Datetime',
    'Altitude based on P0 (FT)',
    'Selected Calibrated Air Speed (KNOTS)',
    'Selected Mach Number (MACH)',
    'Header: Start Date'
]

keys = [
    'selected_ambient_static_pressure_psia',
    'calculated_ambient_temperature_deg_c',
    'selected_total_temperature_at_station_12_deg_c',
    'selected_pt2_pressure_psia',
    'selected_hp_comp_inlet_total_temperature_deg_c',
    'selected_compressor_delay_total_temperature_deg_c',
    'selected_compressor_discharge_static_pressure_psia',
    'selected_exhaust_gas_temperature_deg_c',
    'selected_fan_speed__',
    'selected_core_speed__',
    'selected_mass_fuel_flow_pph',
    'selected_hp_turbine_active_clearance_control_valve_position__',
    'selected_lp_turbine_active_clearance_control_valve_position__',
    'selected_transient_bleed_valve_tbv_position__',
    'selected_variable_bleed_valve_vbv_position__',
    'selected_variable_stator_vane_vsv_position__',
    'total_engine_horsepower_extraction_hp',
    'selected_cai_cowl_anti_ice_bleed_config',
    'selected_booster_anti_ice_bai_pressure_psia',
    'actual_calculated_hpt_clearance_in',
    'synthesized_mass_flow_pps',
    'core_speed_rate_of_change__n2_sec',
    'corrected_fan_speed_to_station_12__',
    'corrected_core_speed_to_station_25__',
    'egt_probe_1_41_deg_cw_alf_deg_c',
    'egt_probe_2_62_deg_cw_alf_deg_c',
    'egt_probe_3_147_deg_cw_alf_deg_c',
    'egt_probe_4_168_deg_cw_alf_deg_c',
    'egt_probe_5_200_deg_cw_alf_deg_c',
    'egt_probe_6_232_deg_cw_alf_deg_c',
    'egt_probe_7_264_deg_cw_alf_deg_c',
    'egt_probe_8_327_deg_cw_alf_deg_c',
    'egt_max_peak_value_deg_c',
    'takeoff_peak_egt_value_deg_c',
    'flight_phase',
    'offset',
    'departure_datetime',
    'altitude',
    'selected_calibrated_air_speed_knots',
    'selected_mach_number_mach',
    'departure_datetime'
]

# Creating the dictionary
convert_CEOD_ZOE = dict(zip(keys, attributes))

# print(convert_CEOD_ZOE)

