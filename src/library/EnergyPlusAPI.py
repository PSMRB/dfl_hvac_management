"""
File to create a class for the EnergyPlus simulation with the API.
"""
from datetime import datetime, timedelta

import numpy as np

from .Building import BuildingModel
from .Common import getcolumnname, get_time, df_linear_interp
from numpy import trunc
import os
import pandas as pd
import sys

energyplus_path = os.environ.get("ENERGYPLUS_PATH")
if energyplus_path:
    sys.path.insert(0, energyplus_path)
else:
    raise RuntimeError("ENERGYPLUS_PATH environment variable not set")
from pyenergyplus.api import EnergyPlusAPI


class EnergyPlusSimulator:
    """
    Class to simulate a building with EnergyPlus using the API. It gathers all the functions necessary to run and
    store the variables as attributes which avoids the use of global variables.
    :param api: the EnergyPlus API
    :param state: the state of the API (an integer)
    :param container: a list to store the variables during the simulation
    :param n_warmup_ts: the number of warm-up time steps (to find and log during the simulation)
    :param run_period_of_interest: the run period of interest (an integer)
    :param day_of_interest: the day of interest (a datetime object) during controlled simulation (actuation)
    :param handles: a dictionnary to store the handles (integer - osrt of position index) of the variables
    :param once: a boolean to check if the handles have been queried (only needed once)
    :param buildingmodel: the building model to simulate
    :param setpoints: the setpoints to actuate
    """

    def __init__(self):
        self.api = None
        self.state = None
        self.container = []
        self.n_warmup_ts = None
        self.run_period_of_interest = None
        self.day_of_interest = None
        self.handles = {}
        self.once = False
        self.buildingmodel = None
        self.setpoints = pd.DataFrame()

    def get_handle(self, state, name):
        """
        Return the handle (i.e., an integer which is a position index) of the variable during the EnergyPlus simulation.
        N.B.: if the handle retrieved is -1, the variable is unknown to Energy+.

        /!\ If using an Output Variable which needs to be requested, please start its name with
        OutputVariable,name (no : and *).

        :param name: the full name of the desired "variables"
        :param type: its type: "InternalVariable", "OutputMeter", "OutputVariable", "Actuator"
        :return:
        """
        api = self.api

        # avoid the semi-colon at the end of the name
        name = name.replace(";", "")

        # split at the first comma
        typ, rest = name.split(',', 1)
        # valid_types = ["int_var", "met", "var", "actuator"]
        # if not any(type == valid_types):
        #     raise NameError(f"value '{type}' for type does not exist.\n"
        #                     f"Valid type value are: {valid_types}")

        if typ == "InternalVariable":
            arg1, arg2 = rest.split(',')
            handle = api.exchange.get_internal_variable_handle(state, arg1, arg2)

        elif typ == "OutputMeter":
            handle = api.exchange.get_meter_handle(state, rest)

        elif typ == "OutputVariable":
            arg1, arg2 = rest.split(',')
            handle = api.exchange.get_variable_handle(state, arg1, arg2)

        elif typ == "Actuator":
            arg1, arg2, arg3 = rest.split(',')
            handle = api.exchange.get_actuator_handle(state, arg1, arg2, arg3)
        else:
            valid_types = ["InternalVariable", "OutputMeter", "OutputVariable", "Actuator"]
            raise TypeError(f"value '{typ}' for type does not exist.\n"
                            f"Valid type value are: {valid_types}")

        if handle == -1:  # The Energy+ API sets the handle to -1 when it did not find it (catch the error)
            raise ValueError(f"The handle of the variable entitled {name} was not found.")
        return rest, (typ, handle)

    def get_value(self, state, typ, handle):
        """
        Return the value of a variable during the EnergyPlus simulation based on its handle.
        :param state:
        :param typ:
        :param handle:
        :return:
        """
        api = self.api

        if typ == "InternalVariable":
            value = api.exchange.get_internal_variable_value(state, handle)
        elif typ == "OutputMeter":
            value = api.exchange.get_meter_value(state, handle)
        elif typ == "OutputVariable":
            value = api.exchange.get_variable_value(state, handle)
        elif typ == "Actuator":
            value = api.exchange.get_actuator_value(state, handle)
        else:
            valid_types = ["InternalVariable", "OutputMeter", "OutputVariable", "Actuator"]
            raise NameError(f"value '{typ}' for type does not exist.\n"
                            f"Valid type value are: {valid_types}")

        return value

    def get_handles(self, state):
        """
        1. Generate the csv of all the variables accessible by the API.
        2. Get the handles of the variables to be recorded/played with during the simulation.
        :param state:
        :return:
        """
        api = self.api

        # print(f"*********** ENVIRONMENT: {api.exchange.current_environment_num(state)}")

        if api.exchange.api_data_fully_ready(state) and not self.once:
            #  Save a list of the actuators (and directly accessible variables and meters) in a csv file
            val = api.exchange.list_available_api_data_csv(state)
            val = val.decode(encoding='utf-8')
            list_val = val.split("\n")
            with open(os.path.join(self.buildingmodel.output_folderpath, 'data.csv'), 'w') as f:
                f.write(val)

            # Get the handle (an integer value) to access the desired (internal) variables, meters and actuators
            # This operation is necessary only once.
            # Their name can be directly copied from the data.csv file created hereabove
            # Fill the list names in the order of which you want the variables to appear in the dataframe
            # N.B. : Per default, timestep is the first variable.
            for n in self.buildingmodel.simulationvariables:
                rest, tup = self.get_handle(state, n)
                self.handles[rest] = tup  # dictionnary to contain the handles

            self.once = True
            # print("Handles queried.\n--------------------")

    def save_variables(self, state):
        """
        save the variables during the simulation in a list.
        :param state:
        :return:
        """
        api = self.api

        # update n_warmup_ts once we are in the warm-up of the yearly simulation (not before during sizing etc.)
        if api.exchange.current_environment_num(state) == self.run_period_of_interest and api.exchange.warmup_flag(
                state):
            self.n_warmup_ts = int(api.exchange.current_sim_time(state)) * self.buildingmodel.simulationfrequency

        # This if saves the data when the timestep is an integer (an hour), therefore all system timesteps are not visible
        # When coding, I advice to record all the system timesteps and then, adjust the recording frequency to your needs
        # Only records during RUN PERIOD (==8) (and not sizing, warmup etc.) To get the index of the RUN PERIOD,
        # open the eplusout.eio file and search "environment," count the number of occurences before arriving at
        # "Environment, RUN PERIOD
        # This section of the code is the one time consuming when runing
        sim_time = api.exchange.current_sim_time(state)
        # for catching hourly values only
        if api.exchange.current_environment_num(state) == self.run_period_of_interest:  # and not warm_up:
            minutes = api.exchange.minutes(state) % 60
            hour = api.exchange.hour(state) + api.exchange.minutes(state) // 60
            day = api.exchange.day_of_month(state)
            month = api.exchange.month(state)

            # Convert 24:00 to 00:00
            if hour == 24:
                hour = 23
                # record the date and time in a datetime object (year arbitrarily set to 2000)
                my_datetime = datetime(2000, month, day,
                                       hour, minutes)
                my_datetime = my_datetime + timedelta(hours=1)
            else:
                # record the date and time in a datetime object (year arbitrarily set to 2000)
                my_datetime = datetime(2000, month, day,
                                       hour, minutes)
            my_datetime = my_datetime.strftime("%m/%d %H:%M")

            ts_data = [sim_time, my_datetime, api.exchange.warmup_flag(state)]
            # t for type, h for handle
            for t, h in self.handles.values():
                ts_data.append(self.get_value(state, t, h))

            self.container.append(ts_data)

    def actuate_setpoints(self, state):
        # used from out of scope but not modified: zone_all_names, electricity_load_schedule, model_COP,
        # fan_electricity_load, api, doi
        api = self.api
        controlled_mask = np.array([("Attic" not in z and "Plenum" not in z) for z in self.buildingmodel.zone_assets.index])
        controlled_zones = self.buildingmodel.zone_assets.loc[controlled_mask]
        uncontrolled_zones = self.buildingmodel.zone_assets.loc[~controlled_mask]

        # update n_warmup_ts once we are in the warm-up of the yearly simulation (not before during sizing etc.)
        if (api.exchange.current_environment_num(state) == self.run_period_of_interest
                and api.exchange.warmup_flag(state)):
            self.n_warmup_ts = int(api.exchange.current_sim_time(state))

        # Actuate only if we are in the RUN PERIOD simulation (i.e., the yearly simulation once sizing is done)
        if (api.exchange.current_environment_num(state) == self.run_period_of_interest
                and not api.exchange.warmup_flag(state)):
            # get the datetime value
            crt_datetime, _ = get_time(api, state)

            # for each zone, actuate the setpoints
            for i, z in enumerate(controlled_zones):
                # Temperature to impose = expected temperature if current time is surrounded by expected results
                if any(dt <= crt_datetime for dt in z.expected_results.index) and any(
                        dt >= crt_datetime for dt in z.expected_results.index):
                    # linear interpolation to get the temperature setpoint
                    setpoint = df_linear_interp(z.expected_results, "Tin", crt_datetime)
                    Tmin = setpoint - 0.05
                    Tmax = setpoint + 0.05
                # else, temperature setpoints are set to historical temperatures
                else:
                    # linear interpolation to get the temperature setpoint
                    setpoint = df_linear_interp(self.buildingmodel.simulation,
                                                f"Zone Mean Air Temperature,{z.name}", crt_datetime)
                    Tmin = setpoint - 0.05
                    Tmax = setpoint + 0.05

                # get the heating setpoint keys
                hk = getcolumnname(f"Zone Temperature Control,Heating Setpoint,{z.name.upper()}",
                                   list(self.handles.keys()))
                # get the cooling setpoint keys
                ck = getcolumnname(f"Zone Temperature Control,Cooling Setpoint,{z.name.upper()}",
                                   list(self.handles.keys()))
                # Get handles to setpoints actuators
                heating_setpoint_actuator = self.handles.get(hk)[1]
                cooling_setpoint_actuator = self.handles.get(ck)[1]

                # Actuate the setpoints
                api.exchange.set_actuator_value(state, heating_setpoint_actuator, Tmin)
                api.exchange.set_actuator_value(state, cooling_setpoint_actuator, Tmax)

            # for i, z in enumerate(uncontrolled_zones):
            #     # No control in the plenums
            #     Tmin = 0
            #     Tmax = 50
            #
            #     # get the heating setpoint keys
            #     hk = getcolumnname(f"Zone Temperature Control,Heating Setpoint,{z.name.upper()}",
            #                        list(self.handles.keys()))
            #     # get the cooling setpoint keys
            #     ck = getcolumnname(f"Zone Temperature Control,Cooling Setpoint,{z.name.upper()}",
            #                        list(self.handles.keys()))
            #     # Get handles to setpoints actuators
            #     heating_setpoint_actuator = self.handles.get(hk)[1]
            #     cooling_setpoint_actuator = self.handles.get(ck)[1]
            #
            #     # Actuate the setpoints
            #     api.exchange.set_actuator_value(state, heating_setpoint_actuator, Tmin)
            #     api.exchange.set_actuator_value(state, cooling_setpoint_actuator, Tmax)

    def callback_no_actuation(self, state):
        self.api.runtime.callback_begin_new_environment(state, self.get_handles)
        self.api.runtime.callback_end_zone_timestep_after_zone_reporting(state, self.save_variables)

    def callback_temperature_control(self, state):
        self.api.runtime.callback_begin_new_environment(state, self.get_handles)
        self.api.runtime.callback_begin_system_timestep_before_predictor(state, self.actuate_setpoints)
        self.api.runtime.callback_end_system_timestep_after_hvac_reporting(state, self.save_variables)

    def run_simulation(self, buildingmodel: BuildingModel, run_period_of_interest: int,
                       callbacks, idf_filepath: str = None):
        """
        Run the simulation of the building model with EnergyPlus using the API. It saves the variables in a dataframe.
        :param buildingmodel:
        :param n_warmup_ts:
        :param run_period_of_interest: an integer indicating the run period of interest
        :param callbacks: a function which gather all the callbacks to be used during the simulation.
        :return:
        """
        self.buildingmodel = buildingmodel
        # dont trust RUN PERIOD, look for "*********** ENVIRONMENT: "
        self.run_period_of_interest = run_period_of_interest

        api = EnergyPlusAPI()
        self.api = api
        state = api.state_manager.new_state()
        # to learn about the various interruption point, see EMSApplicationGuide.pdf which is available in the documentation
        # folder of Energy+ and this page of the GitHub repository:
        # https://github.com/NREL/EnergyPlus/blob/0ff3e33ff6d1d60abcaeb02f021f459840a2faa2/src/EnergyPlus/api/runtime.py
        callbacks(state)
        # impossible to get the variable at fixed hourly time steps
        # api.runtime.callback_end_zone_timestep_after_zone_reporting(state, save_variables)

        # the request_variable function is necessary to access a variable which is not available per default in data.csv list
        # but is used by Energy+. It forces Energy+ to save a variable usually discarded.
        # All the variables used for the simulation are listed in the eplusout.rdd file but not all of them are saved
        # (hence the request_variable function).
        # N.B.: the list of accessible meter is in the eplusout.mdd file (first part of the name) but the second part of the
        # name must be found in data.csv
        for v in buildingmodel.simulationvariables:
            v_nameparts = v.split(",")
            if v_nameparts[0] in ["OutputVariable", "InternalVariable"]:
                api.exchange.request_variable(state, v_nameparts[1], v_nameparts[2])
        # silence output
        api.runtime.set_console_output_status(state, False)
        api.runtime.run_energyplus(state, ["-w", buildingmodel.weather_path, "-d", buildingmodel.output_folderpath,
                                           idf_filepath])

        # stop the simulation and reset the api state
        api.runtime.stop_simulation(state)
        api.runtime.clear_callbacks()
        api.state_manager.delete_state(state)

        container_df = pd.DataFrame(self.container, columns=["sim_time", "Timestep", "warmup"] +
                                                            list(self.handles.keys()))

        return container_df, self.n_warmup_ts
