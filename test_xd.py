from pyomo.environ import *
import numpy as np


def lagrange_ours_m():
    
    
    
    
    
    
    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals)
    model.obj = Objective(expr=(model.x - 3)**2, sense=minimize)
    model.con = Constraint(expr=model.x >= 1)

    solver = SolverFactory('ipopt')
    result = solver.solve(model, tee=True)
    model.pprint()

    solution = np.array([value(model.x)])
    print(solution)