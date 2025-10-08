import itertools
import json
import math
import os
from uuid import uuid4
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import wntr
from wntr.network.controls import _ControlType
import wntr.sim.hydraulics
from wntr.sim.solvers import NewtonSolver
import wntr.sim.results
import warnings
import logging
from wntr.network.base import LinkStatus
from wntr.network.model import WaterNetworkModel
from copy import deepcopy
import plotly.express as px
import plotly.graph_objs as go
import networkx as nx


from wntr.sim.core import _Diagnostics, _ValveSourceChecker, _solver_helper

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

class InteractiveWNTRSimulator(wntr.sim.WNTRSimulator):
    def __init__(self, wn: WaterNetworkModel):
        super().__init__(wn)
        self.initialized_simulation = False
        self._terminated = False
        self._sim_id = f"{uuid4()}"
        self.events_history = []
        self._timestep_index = 0
        self._convergence_error = True

    def hydraulic_timestep(self):
        return self._hydraulic_timestep

    @staticmethod
    def expand_pattern_to_simulation_duration(pattern, time_step=3600, simulation_duration=86400):
        if time_step > simulation_duration:
            raise ValueError('Time step must be less than simulation duration')
        if time_step <= 0:
            raise ValueError('Time step must be greater than zero')
        if pattern is None or len(pattern) == 0:
            raise ValueError('Pattern cannot be None/empty')
        slots = int(simulation_duration / time_step)
        pattern_length = len(pattern)
        new_pattern_length = slots // pattern_length
        new_pattern = []
        for p in pattern:
            new_pattern += [p] * new_pattern_length
        remaining_slots = slots % pattern_length
        if remaining_slots > 0:
            new_pattern.extend([new_pattern[-1]] * remaining_slots)

        return new_pattern

    def init_simulation(self, solver=None, backup_solver=None, solver_options=None, backup_solver_options=None, convergence_error=True, HW_approx='default', diagnostics=False, global_timestep=60, duration=86400):
        logger.debug('creating hydraulic model')
        self.mode = self._wn.options.hydraulic.demand_model
        self._model, self._model_updater = wntr.sim.hydraulics.create_hydraulic_model(wn=self._wn, HW_approx=HW_approx)

        self._hw_approx = HW_approx

        if solver is None:
            solver = NewtonSolver

        if global_timestep is not None:
            self._wn.options.time.hydraulic_timestep = global_timestep
            self._wn.options.time.pattern_timestep = global_timestep 
            self._wn.options.time.rule_timestep = global_timestep
            self._wn.options.time.report_timestep = global_timestep
            self._wn.options.time.quality_timestep = global_timestep

        if duration is not None:
            self._wn.options.time.duration = duration

        self.diagnostics_enabled = diagnostics

        if diagnostics:
            self.diagnostics = _Diagnostics(self._wn, self._model, self.mode, enable=True)
        else:
            self.diagnostics = _Diagnostics(self._wn, self._model, self.mode, enable=False)

        self._setup_sim_options(solver=solver, backup_solver=backup_solver, solver_options=solver_options,
                                backup_solver_options=backup_solver_options, convergence_error=convergence_error)

        self._valve_source_checker = _ValveSourceChecker(self._wn)
        self._get_control_managers()
        self._register_controls_with_observers()

        node_res, link_res = wntr.sim.hydraulics.initialize_results_dict(self._wn)
        
        self.node_res = node_res
        self.link_res = link_res
        
        self.results = wntr.sim.results.SimulationResults()
        self.results.error_code = None
        self.results.time = []
        self.results.network_name = self._wn.name
        self.results.node = {}
        self.results.link = {}

        self._initialize_internal_graph()
        self._change_tracker.set_reference_point('graph')
        self._change_tracker.set_reference_point('model')

        
        self.trial = -1
        self.max_trials = self._wn.options.hydraulic.trials
        self.resolve = False
        self._rule_iter = 0  # this is used to determine the rule timestep

        wntr.sim.hydraulics.update_network_previous_values(self._wn)
        self._wn._prev_sim_time = -1

        self._terminated = False
        self.initialized_simulation = True

        self.last_set_results_time = -1
        
        #self._wn.add_pattern('interactive_pattern', wntrInteractiveSimulator.expand_pattern_to_simulation_duration([1]))

        self.rebuild_hydraulic_model = False
        self.demand_modifications = []
        self.tank_head_modifications = []

        logger.debug('starting simulation')

        logger.info('{0:<10}{1:<10}{2:<10}{3:<15}{4:<15}'.format('Sim Time', 'Trial', 'Solver', '# isolated', '# isolated'))
        logger.info('{0:<10}{1:<10}{2:<10}{3:<15}{4:<15}'.format('', '', '# iter', 'junctions', 'links'))


    def _save_expected_demand_and_leak(self):
        expected_demand = self._model.expected_demand
        for node_name, demand in expected_demand.items():
            v = demand.value
            self.node_res['expected_demand'][node_name].append(v)
            if v == 0:
                self.node_res['satisfied_demand'][node_name].append(1)
            else:
                self.node_res['satisfied_demand'][node_name].append(self.node_res['demand'][node_name][-1] / v)
        
        for node_name, node in self._wn.reservoirs():
            self.node_res['expected_demand'][node_name].append(0.0)
            self.node_res['satisfied_demand'][node_name].append(1.0)
        for node_name, node in self._wn.tanks():
            self.node_res['expected_demand'][node_name].append(0.0)
            self.node_res['satisfied_demand'][node_name].append(1.0)


        leak_rate = self._model.leak_rate
        for node_name, leak in leak_rate.items():
            v = leak.value
            self.node_res['expected_leak'][node_name].append(v)
            if v == 0:
                self.node_res['satisfied_leak'][node_name].append(1)
            else:
                self.node_res['satisfied_leak'][node_name].append(self.node_res['leak_demand'][node_name][-1] / v)  

        for node_name, node in self._wn.reservoirs():
            self.node_res['expected_leak'][node_name].append(0.0)
            self.node_res['satisfied_leak'][node_name].append(1.0)

        #tanks already included in leak_rate    

    def _open_links_graph(self) -> nx.Graph:
        G = nx.Graph()
        for n, _ in self._wn.nodes():
            G.add_node(n)
        for name, link in self._wn.links():
            # only currently open/active links
            if getattr(link, "status", LinkStatus.Open) != LinkStatus.Closed:
                G.add_edge(link.start_node_name, link.end_node_name, key=name)
        return G

    def _unserved_junctions(self):
        G = self._open_links_graph()
        sources = set(self._wn.reservoir_name_list + self._wn.tank_name_list)
        served = set()
        for s in sources:
            if s in G:
                served |= set(nx.node_connected_component(G, s))
        return [j for j in self._wn.junction_name_list if j not in served]

    def _jacobian_info(self):
        try:
            self._model.set_structure()
            J = self._model.evaluate_jacobian(x=None)
            if not sp.sparse.issparse(J):
                return "Jacobian is dense with shape {}".format(np.shape(J))
            nnz = J.nnz
            shape = J.shape
            # crude singularity signal: try spsolve on a random RHS and catch warnings
            try:
                _ = sp.sparse.linalg.spsolve(J.tocsc(), np.ones(shape[0]))
                msg = "Jacobian ok, shape={}, nnz={}".format(shape, nnz)
            except Exception as e:
                msg = "Jacobian solve failed: {} | shape={}, nnz={}".format(e, shape, nnz)
            return msg
        except Exception as e:
            return f"Jacobian diagnostics failed: {e}"

    def _param_sanity(self):
        bad = []
        for name, p in self._wn.pipes():
            if getattr(p, "length", 1.0) <= 0:
                bad.append(f"{name}: length<=0")
            if getattr(p, "diameter", 1.0) <= 0:
                bad.append(f"{name}: diameter<=0")
            C = getattr(p, "roughness", 100.0)
            if C <= 0 or C > 1000:
                bad.append(f"{name}: roughness={C}")
        return bad

    def _log_failure_context(self, mesg: str):
        unserved = self._unserved_junctions()
        if unserved:
            print(f"Time {self._get_time()} — {len(unserved)} junctions disconnected from sources "
                         f"(first 20): {unserved[:20]}")
        else:
            print(f"Time {self._get_time()} — all junctions connected to a source.")

        bad = self._param_sanity()
        if bad:
            print("Suspicious pipe parameters (first 20): " + "; ".join(bad[:20]))
        print("Solver message: " + mesg)
        print(self._jacobian_info())

    def step_sim(self):
        try:
            if not self.initialized_simulation:
                raise RuntimeError('Simulation not initialized. Call init_simulation() before running the simulation.')
            if self._terminated:
                return
            if self._wn.sim_time == 0:
                first_step = True
            else:
                first_step = False

            if not self.resolve:
                if not first_step:
                    """
                    The tank levels/heads must be done before checking the controls because the TankLevelControls
                    depend on the tank levels. These will be updated again after we determine the next actual timestep.
                    """
                    wntr.sim.hydraulics.update_tank_heads(self._wn)
                self.trial = 0
                self._compute_next_timestep_and_run_presolve_controls_and_rules(first_step)

            self._run_feasibility_controls()

            if len(self.demand_modifications) > 0:
                self._apply_demand_modifications()

            if len(self.tank_head_modifications) > 0:
                for tank_name, head in self.tank_head_modifications:
                    tank = self._wn.get_node(tank_name)
                    #head = elevation + level
                    #max_level = head - elevation
                    if tank.max_level < (head - tank.elevation):
                        raise ValueError(f"Tank {tank_name} max level {tank.max_level} is less than {head - tank.elevation}, change max_level before calling init_simulation")
                    tank._head = head
                    tank._prev_head = head
                self.tank_head_modifications.clear()
                

            if self.rebuild_hydraulic_model:
                self._model, self._model_updater = wntr.sim.hydraulics.create_hydraulic_model(wn=self._wn, HW_approx=self._hw_approx)
                self.rebuild_hydraulic_model = False
                
            # Prepare for solve
            self._update_internal_graph()
            isolated_junctions, isolated_links = self._get_isolated_junctions_and_links()
            num_isolated_junctions, num_isolated_links = len(isolated_junctions), len(isolated_links)
            if not first_step and not self.resolve:
                wntr.sim.hydraulics.update_tank_heads(self._wn)
            wntr.sim.hydraulics.update_model_for_controls(self._model, self._wn, self._model_updater, self._change_tracker)
            wntr.sim.models.param.source_head_param(self._model, self._wn)
            wntr.sim.models.param.expected_demand_param(self._model, self._wn)

            
            self.diagnostics.run(last_step='presolve controls, rules, and model updates', next_step='solve')

            solver_status, mesg, iter_count = _solver_helper(self._model, self._solver, self._solver_options)
            if solver_status == 0 and self._backup_solver is not None:
                solver_status, mesg, iter_count = _solver_helper(self._model, self._backup_solver, self._backup_solver_options)
            if solver_status == 0:
                self._log_failure_context(mesg)
                if self._convergence_error:
                    logger.error('Simulation did not converge at time ' + self._get_time() + '. ' + mesg) 
                    raise RuntimeError('Simulation did not converge at time ' + self._get_time() + '. ' + mesg)
                warnings.warn('Simulation did not converge at time ' + self._get_time() + '. ' + mesg)
                logger.warning('Simulation did not converge at time ' + self._get_time() + '. ' + mesg)
                self.results.error_code = wntr.sim.results.ResultsStatus.error
                self.diagnostics.run(last_step='solve', next_step='break')
                self._terminated = True
                self.get_results()
                return

            logger.info('{0:<10}{1:<10}{2:<10}{3:<15}{4:<15}'.format(self._get_time(), self.trial, iter_count, num_isolated_junctions, num_isolated_links))

            # Enter results in network and update previous inputs
            logger.debug('storing results in network')
            wntr.sim.hydraulics.store_results_in_network(self._wn, self._model)

            self.diagnostics.run(last_step='solve and store results in network', next_step='postsolve controls')

            self._run_postsolve_controls()
            self._run_feasibility_controls()
            if self._change_tracker.changes_made(ref_point='graph'):
                self.resolve = True
                self._update_internal_graph()
                wntr.sim.hydraulics.update_model_for_controls(self._model, self._wn, self._model_updater, self._change_tracker)
                self.diagnostics.run(last_step='postsolve controls and model updates', next_step='solve next trial')
                self.trial += 1
                if self.trial > self.max_trials:
                    if self._convergence_error:
                        logger.error('Exceeded maximum number of trials at time ' + self._get_time() + '. ') 
                        raise RuntimeError('Exceeded maximum number of trials at time ' + self._get_time() + '. ' ) 
                    self.results.error_code = wntr.sim.results.ResultsStatus.error
                    warnings.warn('Exceeded maximum number of trials at time ' + self._get_time() + '. ') 
                    logger.warning('Exceeded maximum number of trials at time ' + self._get_time() + '. ' ) 
                    self._terminated = True
                    #self._set_results(self._wn, self.results, self.node_res, self.link_res)
                    return
                self._terminated = False
                #self._set_results(self._wn, self.results, self.node_res, self.link_res)
                return

            self.diagnostics.run(last_step='postsolve controls and model updates', next_step='advance time')

            logger.debug('no changes made by postsolve controls; moving to next timestep')

            self.resolve = False
            if isinstance(self._report_timestep, (float, int)):
                if self._wn.sim_time % self._report_timestep == 0:
                    wntr.sim.hydraulics.save_results(self._wn, self.node_res, self.link_res)
                    self._save_expected_demand_and_leak()

                    if len(self.results.time) > 0 and int(self._wn.sim_time) == self.results.time[-1]:
                        if int(self._wn.sim_time) != self._wn.sim_time:
                            raise RuntimeError('Time steps increments smaller than 1 second are forbidden.'+
                                                ' Keep time steps as an integer number of seconds.')
                        else:
                            raise RuntimeError('Simulation already solved this timestep')
                    self.results.time.append(int(self._wn.sim_time))
            elif self._report_timestep.upper() == 'ALL':
                wntr.sim.hydraulics.save_results(self._wn, self.node_res, self.link_res)
                self._save_expected_demand_and_leak()

                if len(self.results.time) > 0 and int(self._wn.sim_time) == self.results.time[-1]:
                    raise RuntimeError('Simulation already solved this timestep')
                self.results.time.append(int(self._wn.sim_time))
            wntr.sim.hydraulics.update_network_previous_values(self._wn)
            
            self._wn.sim_time += self._hydraulic_timestep
            #overstep = float(self._wn.sim_time) % self._hydraulic_timestep
            #self._wn.sim_time -= overstep
            
            if self._wn.sim_time > self._wn.options.time.duration:
                self._terminated = True
            return
        except Exception as e:
            logger.exception('Error during simulation step at time ' + self._get_time() + '. ' + str(e))
            raise e

    def full_run_sim(self):
        if not self.initialized_simulation:
            raise RuntimeError('Simulation not initialized. Call init_simulation() before running the simulation')
        while not self.is_terminated():
            self.step_sim()
        return
    
    def is_terminated(self):
        return self._terminated
    
    def _set_results(self):
        """
        Ensures all missing timesteps between the last recorded timestep and the current simulation time
        are stored without gaps.
        """

        node_names = self._wn.junction_name_list + self._wn.tank_name_list + self._wn.reservoir_name_list
        link_names = self._wn.pipe_name_list + self._wn.head_pump_name_list + self._wn.power_pump_name_list + self._wn.valve_name_list

        now = self._wn.sim_time 
        last_set_results_time = self.last_set_results_time 

        missing_times = [t for t in self.results.time[self._timestep_index:] if last_set_results_time <= t <= now]

        if not missing_times:
            return 

        for key in self.node_res.keys():
            new_data = np.array([self.node_res[key][name][self._timestep_index:] for name in node_names])
            new_rows = pd.DataFrame(new_data.transpose(), index=missing_times, columns=node_names)
            if key in self.results.node:
                self.results.node[key] = pd.concat([self.results.node[key], new_rows])
            else:
                self.results.node[key] = new_rows

        # Fill in missing link values
        for key in self.link_res.keys():
            new_data = np.array([self.link_res[key][name][self._timestep_index:] for name in link_names])
            new_rows = pd.DataFrame(new_data.transpose(), index=missing_times, columns=link_names)
            if key in self.results.link:
                self.results.link[key] = pd.concat([self.results.link[key], new_rows])
            else:
                self.results.link[key] = new_rows

        self._timestep_index = len(self.results.time)
        self.last_set_results_time = now

    def get_results(self):
        if self.last_set_results_time != self._wn.sim_time:
            self._set_results()
        return self.results
    
    def get_sim_time(self):
        return self._wn.sim_time
    
    def start_leak(self, node_name, leak_area=0.1, leak_discharge_coefficient=0.75):
        self.events_history.append((self.get_sim_time(), 'start_leak', (node_name, leak_area, leak_discharge_coefficient)))

        node = self._wn.get_node(node_name)
        node._leak_status = True
        node._leak = True
        node._leak_area = leak_area
        node._leak_discharge_coeff = leak_discharge_coefficient
        self.rebuild_hydraulic_model = True
        
    def stop_leak(self, node_name):
        self.events_history.append((self.get_sim_time(), 'stop_leak', (node_name)))

        node = self._wn.get_node(node_name)
        node._leak_status = False
        node._leak = False
        self.rebuild_hydraulic_model = True

    def _add_control(self, control):
        if control.epanet_control_type in {_ControlType.presolve, _ControlType.pre_and_postsolve}:
            self._presolve_controls.register_control(control)
        if control.epanet_control_type in {_ControlType.postsolve, _ControlType.pre_and_postsolve}:
            self._postsolve_controls.register_control(control)
        if control.epanet_control_type == _ControlType.rule:
            self._rules.register_control(control)
        if control.epanet_control_type == _ControlType.feasibility:
            self._feasibility_controls.register_control(control)

    def _close_link(self, link_name) -> None:
        link = self._wn.get_link(link_name)
        c1 = wntr.network.controls.ControlAction(link, "status", LinkStatus.Closed)
        condition = wntr.network.controls.SimTimeCondition(self._wn, None, self.get_sim_time()) #+ self.hydraulic_timestep() )
        c = wntr.network.controls.Control(condition=condition, then_action=c1) 
        self._add_control(c)
        self._register_controls_with_observers()
        #link.initial_status = LinkStatus.Closed
        #self.rebuild_hydraulic_model = True

    def _open_link(self, link_name):
        link = self._wn.get_link(link_name)
        c1 = wntr.network.controls.ControlAction(link, "status", LinkStatus.Open)
        condition = wntr.network.controls.SimTimeCondition(self._wn, None, self.get_sim_time()) #+ self.hydraulic_timestep() )
        c = wntr.network.controls.Control(condition=condition, then_action=c1)
        self._add_control(c)
        self._register_controls_with_observers()
        #link.initial_status = LinkStatus.Open
        #self.rebuild_hydraulic_model = True

    def close_pipe(self, pipe_name) -> None:
        self.events_history.append((self.get_sim_time(), 'close_pipe', (pipe_name)))
        self._close_link(pipe_name)
        
    def open_pipe(self, pipe_name):
        self.events_history.append((self.get_sim_time(), 'open_pipe', (pipe_name)))
        self._open_link(pipe_name)
    
    def close_valve(self, valve_name) -> None:
        self.events_history.append((self.get_sim_time(), 'close_valve', (valve_name)))
        self._close_link(valve_name)
        
    def open_valve(self, valve_name):
        self.events_history.append((self.get_sim_time(), 'open_valve', (valve_name)))
        self._open_link(valve_name)
    
    def close_pump(self, pump_name) -> None:
        self.events_history.append((self.get_sim_time(), 'close_pump', (pump_name)))
        self._close_link(pump_name)
           
    def open_pump(self, pump_name):
        self.events_history.append((self.get_sim_time(), 'open_pump', (pump_name)))
        self._open_link(pump_name)

    def set_tank_head(self, tank_name, head):
        self.events_history.append((self.get_sim_time(), 'set_tank_head', (tank_name, head)))
        self.tank_head_modifications.append((tank_name, head))
        #self.rebuild_hydraulic_model = True     

    def set_pump_head_curve(self, pump_name, head_curve):
        self.events_history.append((self.get_sim_time(), 'set_pump_head_curve', (pump_name, head_curve)))
        pump = self._wn.get_link(pump_name)
        if pump.pump_type == 'POWER':
            raise ValueError('Cannot head curve for power pump')
        
        if type(head_curve) == str:
            pump.pump_curve_name = head_curve
        elif type(head_curve) == list[tuple[int, int]]:
            name = f"{pump_name}_Curve_{uuid4()}"
            self._wn.add_curve(name, 'HEAD' , head_curve)  # Head vs. Flow
            pump.pump_curve_name = name

        self.rebuild_hydraulic_model = True
    
    def set_pump_speed_pattern(self, pump_name, speed_pattern):
        self.events_history.append((self.get_sim_time(), 'set_pump_speed_pattern', (pump_name, speed_pattern)))
        pump = self._wn.get_link(pump_name)
        pump.speed_pattern_name = speed_pattern
        self.rebuild_hydraulic_model = True
    
    def set_pump_speed(self, pump_name, speed):
        self.events_history.append((self.get_sim_time(), 'set_pump_speed', (pump_name, speed)))
        pump = self._wn.get_link(pump_name)
        pump.base_speed = speed
        #self.rebuild_hydraulic_model = True

    def plot_network(self, title='Water Network Map', node_labels=True, link_labels=True, figsize=[1920, 1080]):
        wntr.graphics.plot_interactive_network(self._wn, title=f"{title} - {self._sim_id}", node_labels=node_labels, link_labels=link_labels, figsize=figsize)

    def _create_base_figure(self, node_positions, edge_list, node_color_0, edge_color_0,
                        node_hover_0, edge_hover_0, edge_names, node_key, link_key, node_max_value, node_min_value, link_min_value, link_max_value, node_labels=True, link_labels=True):
        # Build node trace (single scatter for all nodes)
        node_x = []
        node_y = []
        node_text = []
        for node, (x, y) in node_positions.items():
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text' if node_labels else 'markers',
            marker=dict(
                size=30,
                color=node_color_0,
                colorscale='viridis_r' if node_key is not None else None,
                colorbar=dict(title=f"{node_key.capitalize()}" if node_key is not None else None),
                cmax=node_max_value if node_key is not None else None,
                cmin=node_min_value if node_key is not None else None,
            ),
            text=node_text,
            hovertext=node_hover_0,
            hoverinfo='text',
            name='Nodes',
            showlegend=False   
        )


        edge_traces = []

        for i, (start, end) in enumerate(edge_list):
            sx, sy = node_positions[start]
            ex, ey = node_positions[end]
            edge_x = [sx, ex]
            edge_y = [sy, ey]

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines+markers',
                line=dict(
                    color=edge_color_0[i], 
                    width=2, 
                ),
                marker=dict(
                    colorscale='Viridis_r' ,
                    colorbar=dict(title=f"{link_key.capitalize()}", x=-0.1 if node_key is not None else None, y = 5 if node_key is not None else None),
                    cmax=link_max_value,
                    cmin=link_min_value,
                ) if link_key is not None and i == 0 else None,  # invisible
                hoverinfo='none',
                name='Edges',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        mid_x = []
        mid_y = []
        mid_hover = []
        annotations = []
        for i, (start, end) in enumerate(edge_list):
            sx, sy = node_positions[start]
            ex, ey = node_positions[end]
            mx, my = (sx+ex)/2, (sy+ey)/2
            mid_x.append(mx)
            mid_y.append(my)
            mid_hover.append(edge_hover_0[i])

            if link_labels:
                annotations.append(dict(
                    x=mx,
                    y=my,
                    xref="x",
                    yref="y",
                    text=edge_names[i],
                    showarrow=False,
                    font=dict(size=10, color='black'),
                    bgcolor="#e5ecf6",
                    align="center"
                ))

        edge_midpoints_trace = go.Scatter(
            x=mid_x,
            y=mid_y,
            mode='markers',
            marker=dict(size=20, color='rgba(0,0,0,0)'),  # invisible
            hoverinfo='text',
            hovertext=mid_hover,
            showlegend=False,
            name='EdgeMidpoints'
        )

        # Build the base layout with the initial annotations

        title = ""
        if node_key is not None:
            title += f"{node_key.capitalize()} "
        if link_key is not None:
            title += f"& {link_key.capitalize()} "
        title += "Over Time"

        layout = go.Layout(
            title=title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode='closest',
            width=1920,
            height=1080,
            #width=1200,
            #height=800,
            annotations=annotations
        )

        # Our base figure data has these 3 traces
        data = []
        data.extend(edge_traces)
        data.append(edge_midpoints_trace)
        data.append(node_trace)

        fig = go.Figure(data=data, layout=layout)
        fig.update_traces(line=dict(width=4))  # Adjust width as needed
        fig.update_layout(
            font=dict(
                # You can choose a font that inherently includes a bold style, such as "Arial Black"
                #family="Arial Black",
                weight="bold",
                size=23  # Make fonts larger
            )
        )

        return fig, annotations

    def _build_frames(self, timesteps, node_color_map, node_hover_map,
                 edge_color_map, edge_hover_map, annotations,
                 num_edges, link_labels=True):
        """
        timesteps: list of times
            e.g. sorted(data_over_time.keys())
        node_color_map[t]: array of node colors at time t
        node_hover_map[t]: array of node hover strings at time t
        edge_color_map[t]: array of edge colors at time t, length = num_edges
        edge_hover_map[t]: array of edge midpoint hover text at time t, length = num_edges
        annotations_map[t]: list of updated annotations for time t
        num_edges: int
            The number of edges (used to slice partial updates)
        """

        frames = []
        for t in timesteps:
            # We'll build a partial data update for each trace.
            # The final data array for each frame must have (num_edges + 2) dicts:
            #   [update_edge_0, update_edge_1, ..., update_edge_(num_edges-1),
            #    update_midpoints, update_nodes]

            frame_data = []

            # 1) Edge traces partial updates
            #    Each edge_color_map[t] is an array of length = num_edges
            #    so we update line.color for each edge trace
            for edge_idx in range(num_edges):
                edge_partial = dict(
                    type='scatter',
                    name='Edges',
                    line=dict(color=edge_color_map[t][edge_idx])
                )
                frame_data.append(edge_partial)

            # 2) Edge midpoints partial update (the next trace after the edges)
            #    We'll set the hovertext for the midpoints to edge_hover_map[t]
            midpoint_partial = dict(
                type='scatter',
                name='EdgeMidpoints',
                hovertext=edge_hover_map[t]
            )
            frame_data.append(midpoint_partial)

            # 3) Node trace partial update (the last trace)
            #    We'll set marker.color and hovertext to node_color_map[t], node_hover_map[t]
            node_partial = dict(
                name='Nodes',
                marker=dict(
                    color=node_color_map[t],
                ),
                hovertext=node_hover_map[t],
            )
            frame_data.append(node_partial)

            # 4) Updated layout with annotations (if your annotations change each frame)
            frame = go.Frame(
                name=str(t),
                data=frame_data,
                layout=go.Layout(
                    annotations=annotations if link_labels else []
                )
            )
            frames.append(frame)

        return frames

    def plot_network_over_time(self, node_key=None, link_key=None, node_labels=True, link_labels=True):
        results = self.get_results()
        node_data_over_time = {}
        edge_data_over_time = {}
        node_hover_map = {}
        edge_hover_map = {}

        node_value_global_maximum = -np.inf
        node_value_global_minimum = np.inf

        link_value_global_maximum = -np.inf
        link_value_global_minimum = np.inf

        for t in results.time:
            node_data_over_time[t] = []
            edge_data_over_time[t] = []
            
            node_hover_map[t] = []
            edge_hover_map[t] = []


            for node in self._wn.nodes(): 
                if node_key is not None and node[1].node_type != "Reservoir" and node[1].node_type != "Tank":
                    v = results.node[node_key][node[0]][t]
                    if v > node_value_global_maximum:
                        node_value_global_maximum = v
                    if v < node_value_global_minimum:
                        node_value_global_minimum = v
                    node_data_over_time[t].append(v)
                else:
                    node_data_over_time[t].append('black')
                
                info = f"Node: {node[0]}"
                for k in results.node.keys():
                    info += f"<br>{k}: {results.node[k][node[0]][t]:.4f}"
                node_hover_map[t].append(info)



            for (name, link) in self._wn.links():
                if link_key is not None:
                    v = results.link[link_key][name][t]
                    if v > link_value_global_maximum:
                        link_value_global_maximum = v
                    if v < link_value_global_minimum:
                        link_value_global_minimum = v
                    edge_data_over_time[t].append(v)
                else:
                    edge_data_over_time[t].append('black')

                start = link.start_node_name
                end = link.end_node_name
                info = f"Edge {name}: {start} - {end}"
                for k in results.link.keys():
                    info += f"<br>{k}: {results.link[k][name][t]:.4f}"
                edge_hover_map[t].append(info)
                #edge_color_map[t].append('black')
            
            if link_key is not None:
                cmap = plt.get_cmap('viridis_r')
                for i, v in enumerate(edge_data_over_time[t]):
                    edge_color_scaled = (v - link_value_global_minimum) / (link_value_global_maximum - link_value_global_minimum)
                    r, g, b, a = cmap(edge_color_scaled)
                    edge_data_over_time[t][i] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"



        node_positions = {}
        for node in self._wn.nodes():
            node_positions[node[0]] = (node[1].coordinates[0], node[1].coordinates[1])

        edges = [(link[1].start_node_name, link[1].end_node_name) for link in self._wn.links()]
        edges_names = [link[0] for link in self._wn.links()]
        times = sorted([t for t in results.time])

        # Build base figure
        fig, annotations = self._create_base_figure(
            node_positions=node_positions,
            edge_list=edges,
            node_color_0=node_data_over_time[times[0]],
            edge_color_0=edge_data_over_time[times[0]],
            node_hover_0=node_hover_map[times[0]],
            edge_hover_0=edge_hover_map[times[0]],
            edge_names=edges_names,
            node_key=node_key,
            link_key=link_key,
            node_max_value=node_value_global_maximum,
            node_min_value=node_value_global_minimum,
            link_max_value=link_value_global_maximum,
            link_min_value=link_value_global_minimum,
            node_labels=node_labels,
            link_labels=link_labels,
        )

        # Build frames
        frames = self._build_frames(
            timesteps=times,
            node_color_map=node_data_over_time,
            node_hover_map=node_hover_map,
            edge_color_map=edge_data_over_time,
            edge_hover_map=edge_hover_map,
            annotations=annotations,
            num_edges=len(edges),
            link_labels=link_labels,
        )

        fig.frames = frames

        # Add standard slider or play button
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=1,
                    x=1.2,
                    xanchor='right',
                    yanchor='top',
                    pad=dict(t=0, r=10),
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[None, dict(
                                frame=dict(duration=250, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode='immediate'
                            )]
                        ),
                        dict(
                            label='Pause',
                            method='animate',
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                                mode='immediate'
                            )]
                        )
                    ]
                )
            ],
            sliders=[dict(
                steps=[
                    dict(
                        label=str(t),
                        method='animate',
                        args=[[str(t)], dict(
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0),
                            mode='immediate'
                        )]
                    ) for t in times
                ]
            )]
        )

        #after = time.time()
        #print(f"Time to build figure: {after - before:.2f} seconds")
        fig.show()

    def add_demand(self, node_name, base_demand, name=None, category=None):
        self.events_history.append((self.get_sim_time(), 'add_demand', (node_name, base_demand, name, category)))
        self.demand_modifications.append(('add', node_name, base_demand, name, category))

    def _apply_demand_modifications(self):
        for action, *args in self.demand_modifications:
            if action == 'add':
                node_name, base_demand, name, category = args
                node = self._wn.get_node(node_name)
                if name is None:
                    node.demand_timeseries_list[0].base_value = base_demand
                else:
                    node._pattern_reg.add_usage(name, (node.name, 'Junction'))
                    node.demand_timeseries_list.append((base_demand, name, category))
            elif action == 'remove':
                node_name, name = args
                if name is None:
                    name = 'interactive_pattern'
                node = self._wn.get_node(node_name)
                node._pattern_reg.remove_usage(name, (node.name, 'Junction'))
                for ts in node.demand_timeseries_list._list:
                    if ts.pattern_name == name:
                        node.demand_timeseries_list.remove(ts)
        self.demand_modifications.clear()

    def start_outage(self, pump_list=None):
        if pump_list is None:
            pump_list = self._wn.pump_name_list
        for pump_name in pump_list:
            self.close_pump(pump_name)
    
    def end_outage(self, pump_list=None):
        if pump_list is None:
            pump_list = self._wn.pump_name_list
        for pump_name in pump_list:
            self.open_pump(pump_name)

    def set_pipe_diameter(self, pipe_name, diameter):
        self.events_history.append((self.get_sim_time(), 'set_pipe_diameter', (pipe_name, diameter)))
        pipe = self._wn.get_link(pipe_name)
        pipe.diameter = diameter
        self.rebuild_hydraulic_model = True

    def remove_demand(self, node_name, name=None):
        self.events_history.append((self.get_sim_time(), 'remove_demand', (node_name, name)))
        self.demand_modifications.append(('remove', node_name, name))

    def change_demand(self, node_name, base_demand=0.0, name=None, category=None):
        node = self._wn.get_node(node_name)
        for ts in node.demand_timeseries_list._list:
            if ts.pattern_name == name:
                self.remove_demand(node_name, name)
                return
        self.add_demand(node_name, base_demand, name, category)    

    def sim_id(self):
        return self._sim_id
    
    def get_stranded_sources(self):
        stranded = []
        for name, _ in itertools.chain(self._wn.tanks(), self._wn.reservoirs()):
            nid = self._node_name_to_id[name]
            rs = self._internal_graph.indptr[nid]
            re = self._internal_graph.indptr[nid + 1]
            if rs == re or not np.any(self._internal_graph.data[rs:re] != 0):
                stranded.append(name)
        return stranded

    def branch(self):
        """
        Creates a branch (copy) of the current simulator instance.

        This method appends a 'create_branch' event to the events history with the current simulation time.
        It then creates a deep copy of the current instance, assigns a new unique simulation ID to the branch,
        and returns the branched instance.

        Returns:
            InteractiveNetworkSimulator: A deep copy of the current simulator instance with a new simulation ID.
        """
        self.events_history.append((self.get_sim_time(), 'create_branch', ()))

        branch = deepcopy(self)
        branch._sim_id = f"{self._sim_id}#branch_{uuid4()}"
        return branch   

    def __deepcopy__(self, memo):
        cls = self.__class__
        # Create a new instance without calling __init__
        result = cls.__new__(cls)
        memo[id(self)] = result
        # Iterate over all attributes in __dict__
        model, model_updater = wntr.sim.hydraulics.create_hydraulic_model(wn=self._wn, HW_approx=self._hw_approx)

        diagnostics = None
        if self.diagnostics_enabled:
            diagnostics = _Diagnostics(self._wn, model, self.mode, enable=True)
        else:
            diagnostics = _Diagnostics(self._wn, model, self.mode, enable=False)

        for key, value in self.__dict__.items():
            if key == "_model":
                setattr(result, key, model)
            elif key == "_model_updater":
                setattr(result, key, model_updater)
            elif key == "diagnostics":
                setattr(result, key, diagnostics)
            else:
                setattr(result, key, deepcopy(value, memo))
        return result

    def plot_results(self, nodeOrLink, key, node_list=None, show_events=True):
        """
        Plots the results of the simulation for a specified key and nodes.
        Parameters:
            nodeOrLink (str): The type of object to plot, either 'node' or 'link'.
            key (str): The key to identify the data to be plotted, can be ['head', 'demand', 'pressure', 'leak_demand'] for nodes, ['flowrate', 'velocity', 'status', 'setting'] for link.
            nodes (list, optional): A list of nodes to filter the data. If None, data for all nodes will be plotted. Defaults to None.
            show_events (bool, optional): If True, events from the events history will be shown on the plot. Defaults to True.
        
        Returns:
            None: This function does not return any value. It displays a plot.
        Notes:
        - The function retrieves the results using the `get_results` method.
        - If no data is available for the specified key and nodes, a message is printed and the function returns.
        - The plot is created using Plotly Express and shows the data over time.
        - If `show_events` is True, vertical lines representing events are added to the plot with different colors and styles based on the event type.
        """


        results = self.get_results()
        data = results.node[key] if nodeOrLink == 'node' else results.link[key]
        data = data[node_list] if node_list is not None else data
        if len(data) == 0:
            print('No results to plot')
            return
        fig = px.line(data, x=data.index, y=data.columns, title=f'{key.capitalize()} at nodes over time - {self._sim_id}')
        fig.update_traces(line=dict(width=4))  # Adjust width as needed
        fig.update_layout(
            font=dict(
                # You can choose a font that inherently includes a bold style, such as "Arial Black"
                #family="Arial Black",
                weight="bold",
                size=23  # Make fonts larger
            ),
            xaxis=dict(
                title=dict(
                    text="Time"
                )
            ),
            yaxis=dict(
                title=dict(
                    text=key.capitalize()
                )
            ),
        )
        if show_events:
            for time, action, args in self.events_history:
                color = '#'+str(hex(hash(action+str(args)) % 16777215)[2:].zfill(6))
                line_style = 'solid' if 'create_branch' in action else 'dash'
                fig.add_vline(x=time, line_dash=line_style, line_color=color, annotation_text='  ', annotation_hovertext=f'{action}{args}', line_width=4)           
        fig.show()

    def _compute_feature_ranges(self, nodes_features, edges_features):
        """Computes min/max for all numerical features before scaling."""
        feature_ranges = {}
        nodes = self._wn.nodes._data.values()
        edges = self._wn.links._data.values()

        for node in nodes:
            for feature in nodes_features:
                value = getattr(node, feature, -1)
                if value is not None and value != -1:
                    if feature not in feature_ranges:
                        feature_ranges[feature] = [value, value]
                    else:
                        feature_ranges[feature][0] = min(feature_ranges[feature][0], value)
                        feature_ranges[feature][1] = max(feature_ranges[feature][1], value)

        for edge in edges:
            for feature in edges_features:
                value = getattr(edge, feature, -1)
                if value is not None and value != -1:
                    if feature not in feature_ranges:
                        feature_ranges[feature] = [value, value]
                    else:
                        feature_ranges[feature][0] = min(feature_ranges[feature][0], value)
                        feature_ranges[feature][1] = max(feature_ranges[feature][1], value)

                    if feature == 'diameter':
                        value = getattr(edge, 'flow', -1)
                        if value is not None and value != -1 and value != 0:
                            velocity = abs(edge.flow)*4.0 / (math.pi*edge.diameter**2)
                            feature = 'velocity'
                            if feature not in feature_ranges:
                                feature_ranges[feature] = [velocity, velocity]
                            else:
                                feature_ranges[feature][0] = min(feature_ranges[feature][0], velocity)
                                feature_ranges[feature][1] = max(feature_ranges[feature][1], velocity)
                
        
        return feature_ranges

    def _scale(self, value, feature, feature_ranges):
        """Scales a feature dynamically based on dataset min/max"""
        has_feature = 0
        if value is None:
            value = -1
        if feature in feature_ranges and value != -1:
            has_feature = 1
            min_val, max_val = feature_ranges[feature]
            if min_val != max_val:  # Avoid division by zero
                value = (value - min_val) / (max_val - min_val)
        
        if abs(value) < 1e-5:
            value = 0
 
        return value, has_feature

    def extract_snapshot(self, scale_values=True, not_scaled_features=[], filename=None):

        results = self.get_results()  # Ensure results are up to date
        nodes = self._wn.nodes._data.values()
        edges = self._wn.links._data.values()

        snapshot = {
            'time': self.get_sim_time(),
            'nodes': {},
            'edges': {},
        }

        node_type_map = {
            'Junction':  [1, 0, 0],
            'Tank':      [0, 1, 0],
            'Reservoir': [0, 0, 1],
        }
        
        edge_type_map = {
            'Pipe':      [1, 0, 0],
            'Pump':      [0, 1, 0],
            'Valve':     [0, 0, 1],
        }

        nodes_features = ['demand', 'elevation', 'head', 'leak_status', 'leak_area',
                        'leak_discharge_coeff', 'leak_demand', 'pressure',
                        'level', 'max_level', 'min_level', 'overflow']
        edges_features = ['base_speed', 'flow', 'headloss', 'roughness', 'velocity', 'diameter']

        nodes_always_valued_features = ["demand", "head", "leak_area", "leak_demand", "leak_discharge_coeff", "leak_status", "pressure"]
        edges_always_valued_features = ['flow']

        feature_ranges = self._compute_feature_ranges(nodes_features, edges_features) if scale_values else {}

        for n in nodes:
            node_data = {}
        
            for feature in nodes_features:
                value = getattr(n, feature, -1)
                
                scaled_value = value
                has_feature = 0
                if scale_values and feature not in not_scaled_features:
                    scaled_value, has_feature = self._scale(value, feature, feature_ranges) 
                elif value is not None and value != -1:
                    has_feature = 1
                else:
                    scaled_value = -1

                node_data[feature] = scaled_value
                if feature not in nodes_always_valued_features:
                    node_data[f'has_{feature}'] = has_feature
                    
            setting = getattr(n, 'setting', -1)
            if isinstance(setting, wntr.network.TimeSeries):
                setting = setting.at(self.get_sim_time())
            if setting is not None and setting != -1:
                node_data['setting'] = setting
                node_data['has_setting'] = 1
            else:
                node_data['setting'] = -1
                node_data['has_setting'] = 0
            
            node_data['expected_demand'] = results.node['expected_demand'][n.name].iloc[-1]
            node_data['satisfied_demand'] = results.node['satisfied_demand'][n.name].iloc[-1]

            node_data['expected_leak'] = results.node['expected_leak'][n.name].iloc[-1]
            node_data['satisfied_leak'] = results.node['satisfied_leak'][n.name].iloc[-1]
            
            node_data['node_type'] = node_type_map[n.node_type]

            snapshot['nodes'][n.name] = node_data

        for l in edges:
            edge_data = {}

            for feature in edges_features:
                value = getattr(l, feature, -1)
                scaled_value, has_feature = self._scale(value, feature, feature_ranges)
                edge_data[feature] = scaled_value
                if feature not in edges_always_valued_features:
                    edge_data[f'has_{feature}'] = has_feature

            setting = getattr(l, 'setting', -1)
            if isinstance(setting, wntr.network.TimeSeries):
                setting = setting.at(self.get_sim_time())
            if setting is not None and setting != -1:
                edge_data['setting'] = setting
                edge_data['has_setting'] = 1
            else:
                edge_data['setting'] = -1
                edge_data['has_setting'] = 0

            if hasattr(l,"diameter") and l.diameter is not None:
                velocity, has_velocity = self._scale(abs(l.flow)*4.0 / (math.pi*l.diameter**2), 'velocity', feature_ranges)
                edge_data["velocity"] = velocity
                edge_data["has_velocity"] = has_velocity
            else:
                edge_data["velocity"] = -1
                edge_data["has_velocity"] = 0
            
            if l.status == LinkStatus.Closed:
                edge_data['status'] = 0
            else:
                edge_data['status'] = 1
            

            edge_data['link_type'] = edge_type_map[l.link_type]  
            edge_data['start'] = l.start_node_name
            edge_data['end'] = l.end_node_name

            snapshot['edges'][l.name] = edge_data

        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(snapshot, f, indent=4, sort_keys=True)

        return snapshot

    def restore_from_snapshot(self, snapshot: dict):
        """
        Restore the hydraulic network *state* from a snapshot created by extract_snapshot(),
        so that the next step continues seamlessly as if the simulation never stopped.

        Requirements:
            - The snapshot must have been produced with scale_value=False.

        Effects:
            - Restores _wn.sim_time to the snapshot time.
            - Restores node-level state: head, pressure, demand, leaks, tank levels, etc.
            - Restores link-level state: status (open/closed), flow, setting/base_speed, etc.
            - Rebuilds the hydraulic model and re-initializes internal solver structures.
            - Syncs 'previous' values so the solver starts from a consistent state.
            - Keeps existing self.results intact and positions indices so new timesteps append.

        Parameters
        ----------
        snapshot : dict
            The dictionary previously returned by extract_snapshot(scale_value=False).
        """
        if not self.initialized_simulation:
            raise RuntimeError("Call init_simulation() before restore_from_snapshot().")

        # ---- 0) Basic checks --------------------------------------------------
        if 'nodes' not in snapshot or 'edges' not in snapshot or 'time' not in snapshot:
            raise ValueError("Malformed snapshot: expected keys ['time','nodes','edges'].")

        def _looks_scaled(node_entry):
            # pressure/head are good canaries; they shouldn't be in [0,1] range for a real network
            for k in ('pressure', 'head'):
                v = node_entry.get(k, None)
                if v is None or v == -1:
                    continue
                if 0.0 <= float(v) <= 1.0:
                    return True
            return False

        any_scaled = any(_looks_scaled(nd) for nd in snapshot['nodes'].values())
        if any_scaled:
            raise ValueError("Snapshot appears scaled. Recreate it with scale_value=False to restore exactly.")

        snap_time = int(snapshot['time'])
        self._wn.sim_time = snap_time

        for name, nd in snapshot['nodes'].items():
            node = self._wn.get_node(name)

            def _set_if_present(attr, key):
                v = nd.get(key, -1)
                if v is not None and v != -1:
                    setattr(node, f"_{attr}", float(v))
                    setattr(node, f"_prev_{attr}", float(v))

            _set_if_present('head', 'head')
            _set_if_present('pressure', 'pressure')
            _set_if_present('demand', 'demand')

            if hasattr(node, 'level'):
                _set_if_present('level', 'level')

            leak_status = nd.get('leak_status', -1)
            if leak_status != -1:
                leak_on = bool(leak_status)
                node._leak_status = leak_on
                node._leak = leak_on
            leak_area = nd.get('leak_area', -1)
            if leak_area != -1:
                node._leak_area = float(leak_area)
            leak_cd = nd.get('leak_discharge_coeff', -1)
            if leak_cd != -1:
                node._leak_discharge_coeff = float(leak_cd)

            setting = nd.get('setting', -1)
            if setting != -1 and hasattr(node, 'setting'):
                try:
                    node.setting = float(setting)
                except Exception:
                    setattr(node, '_setting', float(setting))

        for name, ed in snapshot['edges'].items():
            link = self._wn.get_link(name)

            flow = ed.get('flow', -1)
            if flow != -1:
                link._flow = float(flow)
                link._prev_flow = float(flow)

            status = ed.get('status', None)
            if status is not None:
                if int(status) == 0:
                    link.status = LinkStatus.Closed
                else:
                    link.status = LinkStatus.Open

            setting = ed.get('setting', -1)
            if setting != -1 and hasattr(link, 'setting'):
                try:
                    link.setting = float(setting)
                except Exception:
                    setattr(link, '_setting', float(setting))

            base_speed = ed.get('base_speed', -1)
            if base_speed != -1 and hasattr(link, 'base_speed'):
                link.base_speed = float(base_speed)

            rough = ed.get('roughness', -1)
            if rough != -1 and hasattr(link, 'roughness'):
                link.roughness = float(rough)

            diam = ed.get('diameter', -1)
            if diam != -1 and hasattr(link, 'diameter'):
                link.diameter = float(diam)

        self._model, self._model_updater = wntr.sim.hydraulics.create_hydraulic_model(
            wn=self._wn, HW_approx=self._hw_approx
        )

        if self.diagnostics_enabled:
            self.diagnostics = _Diagnostics(self._wn, self._model, self.mode, enable=True)
        else:
            self.diagnostics = _Diagnostics(self._wn, self._model, self.mode, enable=False)

        self._initialize_internal_graph()
        self._change_tracker.set_reference_point('graph')
        self._change_tracker.set_reference_point('model')

        wntr.sim.hydraulics.update_network_previous_values(self._wn)
        self._wn._prev_sim_time = snap_time - self._hydraulic_timestep

        self.resolve = False
        self._terminated = False

        self.last_set_results_time = snap_time
        self._timestep_index = len(self.results.time)

    @classmethod
    def from_snapshot(cls, wn, snapshot: dict, *, global_timestep: int = 60, duration: int = 24*3600, hw_approx: bool | None = None):
        """
        Build a NEW simulator instance from a snapshot, starting at t=0.
        Assumes 'snapshot' was created by extract_snapshot(scale_value=False).
        'wn' must be a FRESH WaterNetworkModel with the same topology used to create the snapshot.
        """
        sim = cls(wn, hw_approx=hw_approx if hw_approx is not None else getattr(cls, "_hw_approx_default", True))
        sim.init_simulation(global_timestep=global_timestep, duration=duration)

        # 1) Force time origin to 0 (ignore snapshot's time)
        sim._wn.sim_time = 0
        sim._wn._prev_sim_time = -global_timestep  # so next step advances to t=global_timestep

        # 2) Restore node/link state (same logic as restore_from_snapshot, but without results carry-over)
        def _looks_scaled(node_entry):
            for k in ('pressure', 'head'):
                v = node_entry.get(k, None)
                if v is None or v == -1:
                    continue
                if 0.0 <= float(v) <= 1.0:
                    return True
            return False

        any_scaled = any(_looks_scaled(nd) for nd in snapshot['nodes'].values())
        if any_scaled:
            raise ValueError("Snapshot appears scaled. Recreate it with scale_value=False to restore exactly.")

        # ---- Nodes
        for name, nd in snapshot['nodes'].items():
            node = sim._wn.get_node(name)

            def _set_if_present(attr, key):
                v = nd.get(key, -1)
                if v is not None and v != -1:
                    setattr(node, f"_{attr}", float(v))
                    setattr(node, f"_prev_{attr}", float(v))

            _set_if_present('head', 'head')
            _set_if_present('pressure', 'pressure')
            _set_if_present('demand', 'demand')
            if hasattr(node, 'level'):
                _set_if_present('level', 'level')

            leak_status = nd.get('leak_status', -1)
            if leak_status != -1:
                leak_on = bool(leak_status)
                node._leak_status = leak_on
                node._leak = leak_on
            leak_area = nd.get('leak_area', -1)
            if leak_area != -1:
                node._leak_area = float(leak_area)
            leak_cd = nd.get('leak_discharge_coeff', -1)
            if leak_cd != -1:
                node._leak_discharge_coeff = float(leak_cd)

            setting = nd.get('setting', -1)
            if setting != -1 and hasattr(node, 'setting'):
                try:
                    node.setting = float(setting)
                except Exception:
                    setattr(node, '_setting', float(setting))

        # ---- Links
        from wntr.network.controls import LinkStatus  # ensure available
        for name, ed in snapshot['edges'].items():
            link = sim._wn.get_link(name)

            flow = ed.get('flow', -1)
            if flow != -1:
                link._flow = float(flow)
                link._prev_flow = float(flow)

            status = ed.get('status', None)
            if status is not None:
                link.status = LinkStatus.Open if int(status) != 0 else LinkStatus.Closed

            setting = ed.get('setting', -1)
            if setting != -1 and hasattr(link, 'setting'):
                try:
                    link.setting = float(setting)
                except Exception:
                    setattr(link, '_setting', float(setting))

            base_speed = ed.get('base_speed', -1)
            if base_speed != -1 and hasattr(link, 'base_speed'):
                link.base_speed = float(base_speed)

            rough = ed.get('roughness', -1)
            if rough != -1 and hasattr(link, 'roughness'):
                link.roughness = float(rough)

            diam = ed.get('diameter', -1)
            if diam != -1 and hasattr(link, 'diameter'):
                link.diameter = float(diam)

        # 3) Rebuild hydraulic model & sync internals
        import wntr
        sim._model, sim._model_updater = wntr.sim.hydraulics.create_hydraulic_model(
            wn=sim._wn, HW_approx=sim._hw_approx
        )
        sim._initialize_internal_graph()
        sim._change_tracker.set_reference_point('graph')
        sim._change_tracker.set_reference_point('model')

        wntr.sim.hydraulics.update_network_previous_values(sim._wn)

        # 4) Reset diagnostics
        if sim.diagnostics_enabled:
            sim.diagnostics = _Diagnostics(sim._wn, sim._model, sim.mode, enable=True)
        else:
            sim.diagnostics = _Diagnostics(sim._wn, sim._model, sim.mode, enable=False)

        # 5) Start “as new”: clear any residual results and set cursors to 0
        sim.results.clear()                # make sure your Results object has a clear() method; else re-init it
        sim.last_set_results_time = 0
        sim._timestep_index = 0
        sim.resolve = False
        sim._terminated = False

        return sim

    def _calculate_b1(self):
        nodes = self._wn.node_name_list
        links = self._wn.link_name_list

        B1 = np.zeros((len(nodes), len(links)))
        node_idx = {name: i for i, name in enumerate(nodes)}
        link_idx = {name: j for j, name in enumerate(links)}

        for link in links:
            l = self._wn.get_link(link)
            start = l.start_node_name
            end = l.end_node_name

            reversed = False if l.flow >= 0 else True

            if start in node_idx:  # Only consider junctions
                B1[node_idx[start], link_idx[link]] = -1 if not reversed else 1
            if end in node_idx:
                B1[node_idx[end], link_idx[link]] = 1 if not reversed else -1
        
        return B1

    def _calculate_f(self):
        links = self._wn.link_name_list
        results = self.get_results()
        time_index = results.time[-1]
        return np.abs(results.link['flowrate'].loc[time_index, links].values)
    
    def _calculate_d(self):
        nodes = self._wn.node_name_list
        results = self.get_results()
        time_index = results.time[-1]
        return results.node['demand'].loc[time_index, nodes].values
    
    def check_mass_balance(self):
        '''
        B1:
            - N righe (numero di nodi)
            - E colonne (numero di edge)
            - Nella colonna Iesima abbiamo:
                - -1 sul nodo da cui entra l'acqua
                -  1 sul nodo da esce l'acqua 

        F:
            - Vettore colonna in cui abbiamo i flussi nello stesso identico ordine
            delle colonne della matrice B1

        D:
            - le demand che devono avere l'ordine delle righe di B1

        B1 * F - D = matrice tutti 0 -> altrimenti non funziona qualcosa.
        '''
        B1 = self._calculate_b1()
        F = self._calculate_f()
        D = self._calculate_d()
        conservation = B1 @ F - D

        nodes = self._wn.node_name_list
        conservation = pd.Series(conservation, index=nodes)
        filtered = {}
        for k, v in conservation.items():
            if abs(v) > 1e-10:
                filtered[k] = v
        print("Conservation residuals > 1e-10:")
        print(pd.Series(filtered))

        max_error = np.max(np.abs(conservation))
        print(f"Max imbalance at any node: {max_error}")

    def dump_results_to_csv(self):
        results = self.get_results()
        nodes = self._wn.node_name_list
        links = self._wn.link_name_list

        # Create a directory for the results if it doesn't exist
        os.makedirs(f"results/{self._sim_id}", exist_ok=True)

        for key in results.node.keys():
            results.node[key].to_csv(f"results/{self._sim_id}/{self._sim_id}_{key}_nodes.csv", index=True)
        for key in results.link.keys():
            results.link[key].to_csv(f"results/{self._sim_id}/{self._sim_id}_{key}_links.csv", index=True)
        
        header = True
        for event in self.events_history:
            time, action, args = event
            args = str(args)
            args = args.replace("(", '')
            args = args.replace(")", '')
            with open(f"results/{self._sim_id}/{self._sim_id}_events.csv", 'a') as f:
                if header:
                    f.write("time,action,args\n")
                    header = False
                
                f.write(f"{time},{action},{args}\n")