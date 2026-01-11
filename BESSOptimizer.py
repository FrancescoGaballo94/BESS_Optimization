import os 
import numpy as np
import pandas as pd
import math
import pyomo.environ as pyo
import pyomo.opt as po

class BESSOptimizer:
    def __init__(self, solverpath_exe):
        self.solverpath_exe = solverpath_exe  

    def set_glpk_solver(self):
        # Set GLPK solver
        solver = pyo.SolverFactory('glpk', executable=self.solverpath_exe)
        return solver

    def DA_optimization(self, n_cycles, energy_cap: int, power_cap: int, DA_price_vector:list):
        # Calculation of optimal BESS operation for DA market
        # Create a Pyomo model
        model = pyo.ConcreteModel()
        # Number of hours Day-Ahead market
        model.H = pyo.RangeSet(0,23)
        # Number of quarters
        model.Q = pyo.RangeSet(1,96)
        # Number of quarters plus one 
        model.Q_plus_1 = pyo.RangeSet(1,97)
        # Daily discharge limit
        volume_limit = energy_cap * n_cycles
        # Variables definition
        model.soc = pyo.Var(model.Q_plus_1,domain=pyo.Reals)
        model.charge_DA = pyo.Var(model.Q,domain=pyo.NonNegativeReals,bounds=(0,1))
        model.discharge_DA = pyo.Var(model.Q,domain=pyo.NonNegativeReals,bounds=(0,1))
        #Constraints definition
        def set_maximum_soc(model, q):
            return model.soc[q] <= energy_cap
        def set_minimum_soc(model, q):
            return model.soc[q] >= 0
        def set_first_soc(model):
            return model.soc[1] == 0
        def set_soc_balance(model):
            return model.soc[97] == 0
        def soc_step(model, q):
            return model.soc[q+1] == model.soc[q] + (power_cap/4)*model.charge_DA[q] - (power_cap/4)*model.discharge_DA[q]
        def charge_limit(model):
            return sum(model.charge_DA[q]*power_cap/4 for q in model.Q) <= volume_limit
        def discharge_limit(model):
            return sum(model.discharge_DA[q]*power_cap/4 for q in model.Q) <= volume_limit
        ''' Constraints relevant just in case we want to model historical day ahead with hourly resolution 
        def cha_DA_quarters_1_2_parity(model, q):
            return model.charge_DA[4 * q + 1] == model.charge_DA[4 * q + 2]
        def cha_DA_quarters_2_3_parity(model, q):
            return model.charge_DA[4 * q + 2] == model.charge_DA[4 * q + 3]
        def cha_DA_quarters_3_4_parity(model, q):
            return model.charge_DA[4 * q + 3] == model.charge_DA[4 * q + 4]
        def dis_DA_quarters_1_2_parity(model, q):
            return model.discharge_DA[4 * q + 1] == model.discharge_DA[4 * q + 2]
        def dis_DA_quarters_2_3_parity(model, q):
            return model.discharge_DA[4 * q + 2] == model.discharge_DA[4 * q + 3]
        def dis_DA_quarters_3_4_parity(model, q):
            return model.discharge_DA[4 * q + 3] == model.discharge_DA[4 * q + 4]'''
        
        #Application of constraints on the model 
        model.set_maximum_soc = pyo.Constraint(model.Q_plus_1, rule=set_maximum_soc)
        model.set_minimum_soc = pyo.Constraint(model.Q_plus_1, rule=set_minimum_soc)
        model.set_first_soc = pyo.Constraint(rule=set_first_soc)
        model.set_soc_balance = pyo.Constraint(rule=set_soc_balance)
        model.soc_step = pyo.Constraint(model.Q, rule=soc_step)
        model.charge_limit = pyo.Constraint(rule=charge_limit)
        model.discharge_limit = pyo.Constraint(rule=discharge_limit)
        ''' Historical DA with hourly resolution constraints application
        model.cha_DA_quarters_1_2_parity = pyo.Constraint(model.H, rule=cha_DA_quarters_1_2_parity)
        model.cha_DA_quarters_2_3_parity = pyo.Constraint(model.H, rule=cha_DA_quarters_2_3_parity)
        model.cha_DA_quarters_3_4_parity = pyo.Constraint(model.H, rule=cha_DA_quarters_3_4_parity)
        model.dis_DA_quarters_1_2_parity = pyo.Constraint(model.H, rule=dis_DA_quarters_1_2_parity)
        model.dis_DA_quarters_2_3_parity = pyo.Constraint(model.H, rule=dis_DA_quarters_2_3_parity)
        model.dis_DA_quarters_3_4_parity = pyo.Constraint(model.H, rule=dis_DA_quarters_3_4_parity)'''
        # Objective function definition
        def objective_function(model):
            return sum((model.discharge_DA[q] - model.charge_DA[q]) * DA_price_vector[(q)-1] * power_cap/4 for q in model.Q)
        model.objective = pyo.Objective(rule=objective_function, sense=pyo.maximize)
        # Solve the model
        solver = self.set_glpk_solver()
        solver.solve(model)

        # Extract results of SOC, charge, discharge after the day-ahead auction
        step1_soc_DA = [model.soc[q].value for q in range(1,len(DA_price_vector)+1)]
        step1_charge_DA = [model.charge_DA[q].value for q in range(1,len(DA_price_vector)+1)]
        step1_discharge_DA = [model.discharge_DA[q].value for q in range(1,len(DA_price_vector)+1)]

        # Calculation of profit of DA market
        step1_profit_DA = sum(power_cap/4 * (step1_discharge_DA[q] - step1_charge_DA[q]) * DA_price_vector[q] for q in range(0,len(DA_price_vector)))
        
        return(step1_soc_DA, step1_charge_DA, step1_discharge_DA, step1_profit_DA)
        # Intraday optimization 
    
    
    '''Step 2 becomes multi-stage ID auctions:
    for ida_round in ['IDA1', 'IDA2', 'IDA3']:
    optimize_ida(ida_price_vector_round, prior_net_position)
    update_net_position()'''
    
    def ID_Auction_optimization(self, n_cycles: int,energy_cap: int, power_cap: int, IDA_price_vector:list, step1_charge_DA, step1_discharge_DA):

        model=pyo.ConcreteModel()
        # Parameters
        '''historical or if needed forecasted hourly intraday prices 
        model.H = pyo.RangeSet(0,len(IDA_price_vector)/4-1)   '''
        model.Q = pyo.RangeSet(1,len(IDA_price_vector))
        model.Q_plus_1 = pyo.RangeSet(1,len(IDA_price_vector)+1)
        volume_limit = energy_cap * n_cycles
        # Variables definition
        model.soc = pyo.Var(model.Q_plus_1,domain=pyo.Reals)
        model.charge_IDA = pyo.Var(model.Q,domain=pyo.NonNegativeReals, bounds=(0,1))
        model.discharge_IDA = pyo.Var(model.Q,domain=pyo.NonNegativeReals, bounds=(0,1))
        model.charge_closing_IDA = pyo.Var(model.Q,domain=pyo.NonNegativeReals, bounds=(0,1))
        model.discharge_closing_IDA = pyo.Var(model.Q,domain=pyo.NonNegativeReals, bounds=(0,1))
        # Constraints definition
        def set_maximum_soc(model, q):
            return model.soc[q] <= energy_cap
        def set_minimum_soc(model, q):
            return model.soc[q] >= 0
        def set_first_soc(model):
            return model.soc[1] == 0
        def set_soc_balance(model):
            return model.soc[len(IDA_price_vector)+1] == 0
        def soc_step(model, q):
            return model.soc[q+1] == model.soc[q] + (power_cap/4)*(model.charge_IDA[q]-model.discharge_IDA[q]+model.charge_closing_IDA[q]-model.discharge_closing_IDA[q]+step1_charge_DA[q-1]-step1_discharge_DA[q-1])
        def charge_limit(model):
            return (np.sum(step1_charge_DA)+sum(model.charge_IDA[q] for q in model.Q)-sum(model.discharge_closing_IDA[q] for q in model.Q)) *power_cap/4<= volume_limit
        def discharge_limit(model):
            return (np.sum(step1_discharge_DA)+sum(model.discharge_IDA[q] for q in model.Q)-sum(model.charge_closing_IDA[q] for q in model.Q)) *power_cap/4<= volume_limit
        def charge_closing_IDA_logic(model, q):
            return model.charge_closing_IDA[q] <= step1_discharge_DA[q-1]
        def discharge_closing_IDA_logic(model, q):
            return model.discharge_closing_IDA[q] <= step1_charge_DA[q-1]    
        def charge_rate_limit(model, q):
            return model.charge_IDA[q] + step1_charge_DA[q-1] <= 1
        def discharge_rate_limit(model, q):
            return model.discharge_IDA[q] + step1_discharge_DA[q-1] <= 1
        # Application of constraints on the model
        model.set_maximum_soc = pyo.Constraint(model.Q_plus_1, rule=set_maximum_soc)
        model.set_minimum_soc = pyo.Constraint(model.Q_plus_1, rule=set_minimum_soc)
        model.set_first_soc = pyo.Constraint(rule=set_first_soc)
        model.set_soc_balance = pyo.Constraint(rule=set_soc_balance)
        model.soc_step = pyo.Constraint(model.Q, rule=soc_step)
        model.charge_limit = pyo.Constraint(rule=charge_limit)
        model.discharge_limit = pyo.Constraint(rule=discharge_limit)
        model.charge_closing_IDA_logic = pyo.Constraint(model.Q, rule=charge_closing_IDA_logic)
        model.discharge_closing_IDA_logic = pyo.Constraint(model.Q, rule=discharge_closing_IDA_logic)
        model.charge_rate_limit = pyo.Constraint(model.Q, rule=charge_rate_limit)
        model.discharge_rate_limit = pyo.Constraint(model.Q, rule=discharge_rate_limit)
        # Objective function definition
        def objective_function(model):  
            return sum((model.discharge_IDA[q] + model.discharge_closing_IDA[q] - model.charge_IDA[q] - model.charge_closing_IDA[q]) * IDA_price_vector[q-1] * power_cap/4 for q in model.Q)
        model.objective = pyo.Objective(rule=objective_function,sense=pyo.maximize)
        # Solve the model
        solver = self.set_glpk_solver()
        results = solver.solve(model,timelimit=5)
        
        # Check solver status
        if results.solver.status != po.SolverStatus.ok:
            print(f"Solver status: {results.solver.status}")
            print(f"Termination condition: {results.solver.termination_condition}")
            raise Exception("Solver failed to find a solution")
    
        # Extract results of SOC, charge, discharge after the intraday market
        step2_soc_IDA = [model.soc[q].value for q in range(1,len(IDA_price_vector)+1)]
        step2_charge_IDA = [model.charge_IDA[q].value for q in range(1,len(IDA_price_vector)+1)]
        step2_discharge_IDA = [model.discharge_IDA[q].value for q in range(1,len(IDA_price_vector)+1)]
        step2_charge_closing_IDA = [model.charge_closing_IDA[q].value for q in range(1,len(IDA_price_vector)+1)]
        step2_discharge_closing_IDA = [model.discharge_closing_IDA[q].value for q in range(1,len(IDA_price_vector)+1)]
        # Calculation of profit of ID market
        step2_profit_IDA = sum(power_cap/4 * (step2_discharge_IDA[q] + step2_discharge_closing_IDA[q] - step2_charge_IDA[q] - step2_charge_closing_IDA[q]) * IDA_price_vector[q] for q in range(0,len(IDA_price_vector)))
        step2_charge_DA_IDA= np.asarray(step1_charge_DA) - step2_discharge_closing_IDA + step2_charge_IDA
        step2_dischrge_DA_IDA= np.asarray(step1_discharge_DA) - step2_charge_closing_IDA + step2_discharge_IDA
        return(step2_soc_IDA, step2_charge_IDA, step2_discharge_IDA, step2_charge_closing_IDA, step2_discharge_closing_IDA, step2_profit_IDA, step2_charge_DA_IDA, step2_dischrge_DA_IDA)
    
    def ID_Closing_optimization(self,n_cycles:int, energy_cap:int, power_cap:int, IDC_price_vector:list, step2_charge_DA_IDA:list  , step2_discharge_DA_IDA:list):

        model=pyo.ConcreteModel()
        # Parameters
        ''' If hourly intraday prices are used
        model.H = pyo.RangeSet(0,len(IDC_price_vector)/4-1)'''
        model.Q = pyo.RangeSet(1,len(IDC_price_vector))
        model.Q_plus_1 = pyo.RangeSet(1,len(IDC_price_vector)+1)
        volume_limit = energy_cap * n_cycles
        # Variables definition
        model.soc = pyo.Var(model.Q_plus_1,domain=pyo.Reals)
        model.charge_IDC = pyo.Var(model.Q,domain=pyo.NonNegativeReals, bounds=(0,1))
        model.discharge_IDC = pyo.Var(model.Q,domain=pyo.NonNegativeReals, bounds=(0,1))
        model.charge_closing_IDC = pyo.Var(model.Q,domain=pyo.NonNegativeReals, bounds=(0,1))
        model.discharge_closing_IDC = pyo.Var(model.Q,domain=pyo.NonNegativeReals, bounds=(0,1))
        # Constraints definition
        def set_maximum_soc(model, q):
            return model.soc[q] <= energy_cap
        def set_minimum_soc(model, q):
            return model.soc[q] >= 0
        def set_first_soc(model):
            return model.soc[1] == 0
        def set_soc_balance(model):
            return model.soc[len(IDC_price_vector)+1] == 0
        def soc_step(model, q):
            return model.soc[q+1] == model.soc[q] + (power_cap/4)*(model.charge_IDC[q]-model.discharge_IDC[q]+model.charge_closing_IDC[q]-model.discharge_closing_IDC[q]+step2_charge_DA_IDA[q-1]-step2_discharge_DA_IDA[q-1])
        def charge_limit(model):
            return (np.sum(step2_charge_DA_IDA)+sum(model.charge_IDC[q] for q in model.Q)-sum(model.discharge_closing_IDC[q] for q in model.Q)) *power_cap/4<= volume_limit
        def discharge_limit(model):
            return (np.sum(step2_discharge_DA_IDA)+sum(model.discharge_IDC[q] for q in model.Q)-sum(model.charge_closing_IDC[q] for q in model.Q)) *power_cap/4<= volume_limit
        def charge_closing_ID_logic(model, q):
            return model.charge_closing_IDC[q] <= step2_discharge_DA_IDA[q-1]
        def discharge_closing_ID_logic(model, q):
            return model.discharge_closing_IDC[q] <= step2_charge_DA_IDA[q-1]
        def charge_rate_limit(model, q):
            return model.charge_IDC[q] + model.charge_closing_IDC[q] <= 1
        def discharge_rate_limit(model, q):
            return model.discharge_IDC[q] + model.discharge_closing_IDC[q] <= 1
        # Application of constraints on the model 
        model.set_maximum_soc = pyo.Constraint(model.Q_plus_1, rule=set_maximum_soc)
        model.set_minimum_soc = pyo.Constraint(model.Q_plus_1, rule=set_minimum_soc)
        model.set_first_soc = pyo.Constraint(rule=set_first_soc)
        model.set_soc_balance = pyo.Constraint(rule=set_soc_balance)
        model.soc_step = pyo.Constraint(model.Q, rule=soc_step)
        model.charge_limit = pyo.Constraint(rule=charge_limit)
        model.discharge_limit = pyo.Constraint(rule=discharge_limit)
        model.charge_closing_IDC_logic = pyo.Constraint(model.Q, rule=charge_closing_ID_logic)
        model.discharge_closing_IDC_logic = pyo.Constraint(model.Q, rule=discharge_closing_ID_logic)
        model.charge_rate_limit = pyo.Constraint(model.Q, rule=charge_rate_limit)
        model.discharge_rate_limit = pyo.Constraint(model.Q, rule=discharge_rate_limit)
        # Objective function definition
        def objective_function(model):
            return sum((model.discharge_IDC[q] + model.discharge_closing_IDC[q] - model.charge_IDC[q] - model.charge_closing_IDC[q]) * IDC_price_vector[q-1] * power_cap/4 for q in model.Q)
            
        model.objective = pyo.Objective(rule=objective_function,sense=pyo.maximize)
        # Solve the model
        solver = self.set_glpk_solver()
        solver.solve(model,timelimit=5)
        # Extract results of SOC, charge, discharge after the intraday market
        step3_soc_IDC = [model.soc[q].value for q in range(1,len(IDC_price_vector)+1)]
        step3_charge_IDC = [model.charge_IDC[q].value for q in range(1,len(IDC_price_vector)+1)]
        step3_discharge_IDC = [model.discharge_IDC[q].value for q in range(1,len(IDC_price_vector)+1)]
        step3_charge_closing_IDC = [model.charge_closing_IDC[q].value for q in range(1,len(IDC_price_vector)+1)]
        step3_discharge_closing_IDC = [model.discharge_closing_IDC[q].value for q in range(1,len(IDC_price_vector)+1)]
        # Calculation of profit of ID market
        step3_profit_IDC = np.sum(power_cap/4 * (np.asarray(step3_discharge_IDC) + step3_discharge_closing_IDC - np.asarray(step3_charge_IDC) - step3_charge_closing_IDC) * IDC_price_vector)
        step3_cha_DA_IDA_IDC= np.asarray(step2_charge_DA_IDA) - step3_discharge_closing_IDC + step3_charge_IDC
        step3_dis_DA_IDA_IDC= np.asarray(step2_discharge_DA_IDA) - step3_charge_closing_IDC + step3_discharge_IDC
        return(step3_soc_IDC, step3_charge_IDC, step3_discharge_IDC, step3_charge_closing_IDC, step3_discharge_closing_IDC, step3_profit_IDC, step3_cha_DA_IDA_IDC, step3_dis_DA_IDA_IDC)



    
    